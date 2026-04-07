"""
Scene Graph Matching for SG-Nav
================================
Match two independently-constructed scene graphs from different exploration runs.
Detects object correspondences and scene changes (moved, removed, added objects).

Usage:
    python -m tools.sg_match \
        --sg1 data/usefulruns/1-global_sg_goal_chair_1_found_20260316_232325.pkl \
        --sg2 data/usefulruns/2-global_sg_goal_table_2_found_20260316_235307.pkl \
        --verbose

    python tools/sg_match.py --sg1 <path1.pkl> --sg2 <path2.pkl> --output result.json
"""

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def _ts():
    """Compact timestamp for progress messages."""
    return time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ChangeType(str, Enum):
    SAME = "SAME"
    MOVED = "MOVED"
    REPLACED = "REPLACED"
    ADDED = "ADDED"
    REMOVED = "REMOVED"
    UNSEEN = "UNSEEN"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class NodeFeatures:
    """Extracted matching features for a single node."""
    idx: int                         # index in the serialized node list
    caption: str
    clip_ft: np.ndarray              # (512,) CLIP visual embedding
    text_ft: np.ndarray              # (512,) CLIP text embedding
    centroid_3d: np.ndarray          # (3,) mean of point cloud
    center_2d: Optional[list]        # [x, y] map grid coords
    room_idx: Optional[int]
    num_detections: int
    num_points: int                  # total points in point cloud
    bbox_volume: float               # oriented bounding box volume
    color_hist: Optional[np.ndarray] = None   # (48,) HSV histogram
    shape_desc: Optional[np.ndarray] = None   # (4,) PCA eigenratios + log-volume
    neighbor_cat_hist: Optional[np.ndarray] = None  # (C,) neighbor category histogram
    neighbor_clip_agg: Optional[np.ndarray] = None  # (512,) mean neighbor CLIP


@dataclass
class MatchResult:
    """Result of matching a single node pair."""
    sg1_idx: int
    sg2_idx: int
    sg1_caption: str
    sg2_caption: str
    change_type: str
    cost: float
    clip_visual_sim: float
    spatial_dist_3d: float
    clip_text_sim: float
    color_hist_sim: float
    same_room: bool
    edge_consistency: float
    confidence: float


@dataclass
class UnmatchedNode:
    """A node that has no correspondence in the other graph."""
    sg_source: str                   # "sg1" or "sg2"
    node_idx: int
    caption: str
    change_type: str                 # ADDED or REMOVED
    centroid_3d: list
    room_idx: Optional[int]


@dataclass
class MatchReport:
    """Full matching report between two scene graphs."""
    sg1_path: str
    sg2_path: str
    sg1_num_nodes: int
    sg2_num_nodes: int
    sg1_num_edges: int
    sg2_num_edges: int
    matches: list
    unmatched: list
    edge_consistency_global: float
    summary: dict


# ---------------------------------------------------------------------------
# Spatial utilities
# ---------------------------------------------------------------------------

MAP_RESOLUTION = 0.05   # meters per pixel (SG-Nav default)
MAP_SIZE = 800           # grid dimension

def _3d_to_grid(centroid_3d: np.ndarray) -> Tuple[int, int]:
    """Convert 3D world coords to (col, row) in the 800×800 grid."""
    col = int(round(centroid_3d[0] / MAP_RESOLUTION))
    row = int(round(MAP_SIZE - centroid_3d[1] / MAP_RESOLUTION))
    return col, row


def compute_auto_scene_scale(feats_1: list, feats_2: list) -> float:
    """Compute scene_scale from the union bounding box of both graphs.

    Returns the max spatial extent (in meters) across X/Y/Z.
    Falls back to 20.0 if insufficient data.
    """
    all_centroids = []
    for f in feats_1:
        all_centroids.append(f.centroid_3d)
    for f in feats_2:
        all_centroids.append(f.centroid_3d)
    if len(all_centroids) < 2:
        return 20.0
    pts = np.array(all_centroids)
    extents = pts.max(axis=0) - pts.min(axis=0)
    scale = float(extents.max())
    return max(scale, 1.0)


def build_explored_mask(maps: dict, sensor_range_m: float = 5.0) -> Optional[np.ndarray]:
    """Build a boolean mask of explored/observable cells from maps data.

    Combines the visited map and free-space map, then expands by the
    sensor range so that objects visible *from* visited cells are covered.
    Returns an (800, 800) boolean array, or None if maps unavailable.
    """
    visited = maps.get('global_visited')
    free_map = maps.get('global_fbe_free_map')

    if visited is None:
        return None

    visited_2d = np.asarray(visited).squeeze()
    if visited_2d.ndim != 2:
        return None

    # Combine: a cell is "observable" if it was visited OR detected as free
    observable = visited_2d > 0
    if free_map is not None:
        free_2d = np.asarray(free_map).squeeze()
        if free_2d.ndim == 2 and free_2d.shape == observable.shape:
            observable = observable | (free_2d > 0)

    # Expand by sensor range using distance transform (fast, O(n))
    dilation_px = int(sensor_range_m / MAP_RESOLUTION)
    if dilation_px > 0:
        from scipy.ndimage import distance_transform_edt
        # distance_transform gives distance from each False cell to nearest True
        dist = distance_transform_edt(~observable)
        observable = dist <= dilation_px

    return observable


def is_in_explored_area(centroid_3d: np.ndarray,
                        explored_mask: Optional[np.ndarray]) -> bool:
    """Check whether a 3D point falls within an explored/observable region."""
    if explored_mask is None:
        return True  # no map → assume explored (conservative)
    col, row = _3d_to_grid(centroid_3d)
    if 0 <= row < explored_mask.shape[0] and 0 <= col < explored_mask.shape[1]:
        return bool(explored_mask[row, col])
    return False


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Fast vectorized RGB→HSV.  Input: (N, 3) floats in [0,1]."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delta = maxc - minc
    with np.errstate(invalid='ignore', divide='ignore'):
        s = np.where(maxc > 1e-8, delta / maxc, 0.0)
    # hue
    h = np.zeros_like(r)
    mask = delta > 1e-8
    idx_r = mask & (maxc == r)
    idx_g = mask & (maxc == g) & ~idx_r
    idx_b = mask & ~idx_r & ~idx_g
    h[idx_r] = ((g[idx_r] - b[idx_r]) / delta[idx_r]) % 6
    h[idx_g] = (b[idx_g] - r[idx_g]) / delta[idx_g] + 2
    h[idx_b] = (r[idx_b] - g[idx_b]) / delta[idx_b] + 4
    h = h / 6.0
    h[h < 0] += 1.0
    return np.stack([h, s, v], axis=1)


def _compute_color_histogram(pcd_colors: np.ndarray, bins: int = 16) -> Optional[np.ndarray]:
    """Compute a normalized HSV histogram from point cloud colors.

    Returns a (3*bins,) float32 vector, or None if too few points.
    """
    if pcd_colors is None or len(pcd_colors) < 10:
        return None
    colors = np.asarray(pcd_colors, dtype=np.float64)
    if colors.max() > 1.0:
        colors = colors / 255.0
    colors = np.clip(colors, 0.0, 1.0)
    hsv = _rgb_to_hsv(colors)
    hist = np.empty(3 * bins, dtype=np.float32)
    for ch in range(3):
        h, _ = np.histogram(hsv[:, ch], bins=bins, range=(0.0, 1.0))
        hist[ch * bins:(ch + 1) * bins] = h
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _compute_shape_descriptor(pcd_points: np.ndarray,
                              bbox_points: np.ndarray) -> Optional[np.ndarray]:
    """PCA eigenvalue ratios + log-volume shape descriptor.

    Returns a (4,) float32 vector, or None if too few points.
    """
    if pcd_points is None or len(pcd_points) < 10:
        return None
    centered = pcd_points - pcd_points.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigvals = np.maximum(eigvals, 1e-10)
    total = eigvals.sum()
    ratios = (eigvals / total).astype(np.float32)
    extents = pcd_points.max(axis=0) - pcd_points.min(axis=0)
    log_vol = np.float32(np.log(np.prod(np.maximum(extents, 0.01)) + 1e-6))
    return np.concatenate([ratios, [log_vol]])


def extract_node_features(sg_data: dict, label: str = "SG") -> List[NodeFeatures]:
    """Extract matching features from a serialized scene graph dict."""
    features = []
    total = len(sg_data['nodes'])
    print(f"[{_ts()}] Extracting features for {label} ({total} raw nodes) ...")
    skipped = 0
    for idx, s_node in enumerate(sg_data['nodes']):
        obj = s_node.get('object')
        if obj is None:
            continue

        clip_ft = np.array(obj.get('clip_ft', []), dtype=np.float32)
        text_ft = np.array(obj.get('text_ft', []), dtype=np.float32)

        if clip_ft.size == 0 or text_ft.size == 0:
            continue

        pcd_points = np.array(obj.get('pcd_points', []), dtype=np.float32)
        if len(pcd_points) == 0:
            continue
        centroid_3d = pcd_points.mean(axis=0)

        bbox_points = np.array(obj.get('bbox_points', []), dtype=np.float32)
        if len(bbox_points) >= 2:
            extents = bbox_points.max(axis=0) - bbox_points.min(axis=0)
            bbox_volume = float(np.prod(extents))
        else:
            bbox_volume = 0.0

        num_detections = obj.get('num_detections', 1)
        if isinstance(num_detections, list):
            num_detections = sum(num_detections)

        pcd_colors = np.array(obj.get('pcd_colors', []), dtype=np.float64)
        color_hist = _compute_color_histogram(pcd_colors)
        shape_desc = _compute_shape_descriptor(pcd_points, bbox_points)

        features.append(NodeFeatures(
            idx=idx,
            caption=s_node.get('caption', ''),
            clip_ft=clip_ft,
            text_ft=text_ft,
            centroid_3d=centroid_3d,
            center_2d=s_node.get('center'),
            room_idx=s_node.get('room_idx'),
            num_detections=num_detections,
            num_points=len(pcd_points),
            bbox_volume=bbox_volume,
            color_hist=color_hist,
            shape_desc=shape_desc,
        ))
    print(f"[{_ts()}]   -> {len(features)} valid nodes extracted "
          f"({total - len(features)} skipped)")
    return features


def build_node_edge_snapshots(sg_data: dict, label: str = "SG") -> Dict[int, np.ndarray]:
    """Build a map: node_idx -> (K, 512) matrix of L2-normalized edge snapshot CLIPs."""
    print(f"[{_ts()}] Building edge snapshots for {label} ...")
    raw: Dict[int, list] = defaultdict(list)
    for e in sg_data.get('edges', []):
        scf = e.get('snapshot_clip_features')
        if scf is None:
            continue
        feat = np.asarray(scf, dtype=np.float32).flatten()
        if feat.shape[0] != 512:
            continue
        norm = np.linalg.norm(feat)
        if norm < 1e-8:
            continue
        feat_norm = feat / norm
        raw[e['node1_idx']].append(feat_norm)
        raw[e['node2_idx']].append(feat_norm)

    result = {nidx: np.stack(fts) for nidx, fts in raw.items() if fts}
    print(f"[{_ts()}]   -> {len(result)} nodes with snapshots")
    return result


def build_adjacency(sg_data: dict) -> Dict[int, List[int]]:
    """Build adjacency lists: node_idx -> sorted list of neighbor node_idx."""
    adj: Dict[int, set] = defaultdict(set)
    for e in sg_data.get('edges', []):
        n1, n2 = e['node1_idx'], e['node2_idx']
        adj[n1].add(n2)
        adj[n2].add(n1)
    return {k: sorted(v) for k, v in adj.items()}


def compute_neighborhood_features(feats: List[NodeFeatures],
                                  adj: Dict[int, List[int]],
                                  cat_to_bin: Dict[str, int],
                                  label: str = "SG") -> None:
    """Fill in neighbor_cat_hist and neighbor_clip_agg for each node in-place.

    neighbor_cat_hist: normalized histogram over neighbor categories
    neighbor_clip_agg: mean L2-normalized CLIP embedding of neighbors

    cat_to_bin must be a shared vocabulary across both graphs so that
    histograms are directly comparable.
    """
    print(f"[{_ts()}] Computing neighborhood features for {label} ...")
    idx_to_feat: Dict[int, NodeFeatures] = {f.idx: f for f in feats}
    n_cats = len(cat_to_bin)

    n_with_neighbors = 0
    for f in feats:
        neighbors = adj.get(f.idx, [])
        neighbor_feats = [idx_to_feat[n] for n in neighbors if n in idx_to_feat]

        if not neighbor_feats:
            f.neighbor_cat_hist = np.zeros(n_cats, dtype=np.float32)
            f.neighbor_clip_agg = np.zeros_like(f.clip_ft)
            continue

        n_with_neighbors += 1

        # Category histogram (shared vocabulary)
        hist = np.zeros(n_cats, dtype=np.float32)
        for nf in neighbor_feats:
            cat = nf.caption.lower().strip().replace('_', ' ')
            if cat in cat_to_bin:
                hist[cat_to_bin[cat]] += 1.0
        total = hist.sum()
        if total > 0:
            hist /= total
        f.neighbor_cat_hist = hist

        # Mean CLIP aggregate
        clip_stack = np.stack([nf.clip_ft for nf in neighbor_feats])
        mean_clip = clip_stack.mean(axis=0)
        norm = np.linalg.norm(mean_clip)
        if norm > 1e-8:
            mean_clip /= norm
        f.neighbor_clip_agg = mean_clip.astype(np.float32)

    print(f"[{_ts()}]   -> {n_with_neighbors}/{len(feats)} nodes with ≥1 neighbor, "
          f"{n_cats} shared category bins")


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, handling zero vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _label_similarity(cap_a: str, cap_b: str) -> float:
    """Label similarity: 1.0 exact, 0.7 substring, 0.0 different."""
    a = cap_a.lower().strip().replace('_', ' ')
    b = cap_b.lower().strip().replace('_', ' ')
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.7
    return 0.0


def clip_visual_cost(a: NodeFeatures, b: NodeFeatures) -> float:
    return 1.0 - _cosine_sim(a.clip_ft, b.clip_ft)


def clip_text_cost(a: NodeFeatures, b: NodeFeatures) -> float:
    return 1.0 - _cosine_sim(a.text_ft, b.text_ft)


def spatial_cost(a: NodeFeatures, b: NodeFeatures,
                 scene_scale: float = 5.0) -> float:
    """Gaussian spatial cost — sharp falloff penalizes distant matches.

    Uses 1 - exp(-d²/2σ²) with σ capped so that:
    - Within ~2m is cheap (true matches with SLAM drift)
    - Beyond ~5m is expensive (likely wrong matches)
    Sigma is min(scene_scale/2.5, 4.0) to prevent the curve from flattening
    too much on large scenes.
    """
    dist = float(np.linalg.norm(a.centroid_3d - b.centroid_3d))
    sigma = min(scene_scale / 2.5, 4.0)
    return 1.0 - float(np.exp(-(dist ** 2) / (2 * sigma ** 2)))


def spatial_dist_meters(a: NodeFeatures, b: NodeFeatures) -> float:
    """Raw 3D Euclidean distance in meters."""
    return float(np.linalg.norm(a.centroid_3d - b.centroid_3d))


def label_cost(a: NodeFeatures, b: NodeFeatures) -> float:
    ca = a.caption.lower().strip().replace('_', ' ')
    cb = b.caption.lower().strip().replace('_', ' ')
    if ca == cb:
        return 0.0
    if ca in cb or cb in ca:
        return 0.3
    return 1.0


def room_cost(a: NodeFeatures, b: NodeFeatures) -> float:
    if a.room_idx is None or b.room_idx is None:
        return 0.5
    return 0.0 if a.room_idx == b.room_idx else 1.0


def snapshot_context_cost(snaps_a: Optional[np.ndarray],
                          snaps_b: Optional[np.ndarray]) -> float:
    """Cost based on best-matching edge snapshot CLIP between two nodes.

    For each node we have a matrix of (K, 512) normalized snapshot features.
    The best-match similarity is max(A @ B^T).  Returns 1 - max_sim.
    If either node has no snapshots, returns 0.5 (no info).
    """
    if snaps_a is None or snaps_b is None:
        return 0.5
    sim_matrix = snaps_a @ snaps_b.T       # (Ka, Kb)
    return 1.0 - float(sim_matrix.max())


def color_hist_cost(a: NodeFeatures, b: NodeFeatures) -> float:
    """Cost from HSV color histogram intersection.

    Histogram intersection is the sum of element-wise minimums of two
    normalized histograms.  Returns 1 - intersection (0 = identical colors).
    """
    if a.color_hist is None or b.color_hist is None:
        return 0.5
    intersection = float(np.minimum(a.color_hist, b.color_hist).sum())
    return 1.0 - intersection


def shape_cost(a: NodeFeatures, b: NodeFeatures) -> float:
    """Cost from PCA-based shape descriptor similarity (cosine)."""
    if a.shape_desc is None or b.shape_desc is None:
        return 0.5
    return 1.0 - _cosine_sim(a.shape_desc, b.shape_desc)


def neighbor_category_cost(a: NodeFeatures, b: NodeFeatures) -> float:
    """Cost from neighborhood category histogram intersection.

    Compares what types of objects surround each node. Position-independent:
    if a chair is next to [table, lamp, rug] in both graphs, cost is low
    even if the chair has moved.  Returns 0.5 if either node has no histogram.
    """
    if a.neighbor_cat_hist is None or b.neighbor_cat_hist is None:
        return 0.5
    intersection = np.minimum(a.neighbor_cat_hist, b.neighbor_cat_hist).sum()
    return 1.0 - float(intersection)


def neighbor_clip_cost(a: NodeFeatures, b: NodeFeatures) -> float:
    """Cost from mean-neighbor CLIP aggregate similarity.

    Compares the mean CLIP visual embedding of 1-hop neighbors.
    More discriminative than category histogram alone — captures visual
    differences between neighborhoods with same category distribution.
    """
    if a.neighbor_clip_agg is None or b.neighbor_clip_agg is None:
        return 0.5
    return 1.0 - _cosine_sim(a.neighbor_clip_agg, b.neighbor_clip_agg)


# ---------------------------------------------------------------------------
# Cost matrix & Hungarian assignment
# ---------------------------------------------------------------------------

@dataclass
class MatchWeights:
    """Weights for combining cost components (10 components).

    Two presets available via `MatchWeights.preset()`:
      - "static"  — spatial position is primary signal (objects don't move)
      - "dynamic" — identity + neighborhood is primary signal (objects may move)
    """
    w_spatial: float = 0.22
    w_color: float = 0.16
    w_nch: float = 0.12
    w_clip_visual: float = 0.12
    w_neighbor_clip: float = 0.08
    w_snapshot: float = 0.08
    w_label: float = 0.07
    w_shape: float = 0.06
    w_clip_text: float = 0.04
    w_room: float = 0.05
    unmatched_cost: float = 0.42
    scene_scale: float = 5.0

    @staticmethod
    def preset(mode: str) -> 'MatchWeights':
        if mode == "static":
            # Position-dependent: 27%, Identity+neighborhood: 73%
            return MatchWeights(
                w_spatial=0.22, w_color=0.16, w_nch=0.12,
                w_clip_visual=0.12, w_neighbor_clip=0.08,
                w_snapshot=0.08, w_label=0.07, w_shape=0.06,
                w_clip_text=0.04, w_room=0.05,
                unmatched_cost=0.42)
        elif mode == "dynamic":
            # Position-dependent: 9%, Identity+neighborhood: 91%
            # NCH is crucial — position-independent structural context
            return MatchWeights(
                w_spatial=0.05, w_color=0.20, w_nch=0.18,
                w_clip_visual=0.14, w_neighbor_clip=0.12,
                w_snapshot=0.10, w_label=0.08, w_shape=0.05,
                w_clip_text=0.04, w_room=0.04,
                unmatched_cost=0.38)
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'static' or 'dynamic'.")


def build_cost_matrix(feats_1: list, feats_2: list,
                      weights: MatchWeights = None,
                      snaps_1: Dict[int, np.ndarray] = None,
                      snaps_2: Dict[int, np.ndarray] = None) -> np.ndarray:
    """Build the padded cost matrix for Hungarian assignment.

    Returns an (n1+n2) x (n1+n2) matrix where dummy entries have
    cost = weights.unmatched_cost.
    """
    if weights is None:
        weights = MatchWeights()

    n1, n2 = len(feats_1), len(feats_2)
    n = n1 + n2

    cost = np.full((n, n), weights.unmatched_cost, dtype=np.float64)

    total_cells = n1 * n2
    report_interval = max(1, n1 // 10)
    t0 = time.time()
    print(f"[{_ts()}] Building cost matrix ({n1} x {n2} = {total_cells:,} cells) ...")

    for i, fa in enumerate(feats_1):
        if report_interval >= 5 and i % report_interval == 0 and i > 0:
            elapsed = time.time() - t0
            pct = 100 * i / n1
            eta = elapsed / i * (n1 - i)
            print(f"[{_ts()}]   {pct:5.1f}% ({i}/{n1} rows, "
                  f"{elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)")
        sa = snaps_1.get(fa.idx) if snaps_1 else None
        for j, fb in enumerate(feats_2):
            sb = snaps_2.get(fb.idx) if snaps_2 else None
            c_vis = clip_visual_cost(fa, fb)
            c_spa = spatial_cost(fa, fb, weights.scene_scale)
            c_lbl = label_cost(fa, fb)
            c_txt = clip_text_cost(fa, fb)
            c_rm = room_cost(fa, fb)
            c_snap = snapshot_context_cost(sa, sb)
            c_color = color_hist_cost(fa, fb)
            c_shape = shape_cost(fa, fb)
            c_nch = neighbor_category_cost(fa, fb)
            c_nclip = neighbor_clip_cost(fa, fb)

            cost[i, j] = (weights.w_clip_visual * c_vis +
                          weights.w_spatial * c_spa +
                          weights.w_label * c_lbl +
                          weights.w_clip_text * c_txt +
                          weights.w_room * c_rm +
                          weights.w_snapshot * c_snap +
                          weights.w_color * c_color +
                          weights.w_shape * c_shape +
                          weights.w_nch * c_nch +
                          weights.w_neighbor_clip * c_nclip)

    elapsed = time.time() - t0
    print(f"[{_ts()}]   Cost matrix built in {elapsed:.1f}s")
    return cost


def solve_assignment(cost_matrix: np.ndarray, n1: int, n2: int,
                     unmatched_cost: float) -> list:
    """Solve the assignment and return list of (i, j, cost) tuples."""
    n = cost_matrix.shape[0]
    print(f"[{_ts()}] Solving Hungarian assignment ({n}x{n} matrix) ...")
    t0 = time.time()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    elapsed = time.time() - t0
    print(f"[{_ts()}]   Solved in {elapsed:.1f}s")

    pairs = []
    for r, c in zip(row_ind, col_ind):
        if r < n1 and c < n2:
            pairs.append((r, c, cost_matrix[r, c]))
    return pairs


def iterative_neighborhood_consensus(
        cost_matrix: np.ndarray,
        feats_1: List[NodeFeatures], feats_2: List[NodeFeatures],
        adj_1: Dict[int, List[int]], adj_2: Dict[int, List[int]],
        n_iters: int = 3,
        bonus: float = 0.04, penalty: float = 0.03,
        unmatched_cost: float = 0.42) -> list:
    """Iterative refinement: adjust costs based on neighborhood match consistency.

    After initial Hungarian, for each matched pair (a→b):
      - Count what fraction of a's neighbors are matched to b's neighbors
      - High consistency → reduce cost (bonus); low → increase cost (penalty)
    Re-solve and repeat for n_iters rounds.

    This enforces structural consistency: if chair-A matches chair-X, then
    chair-A's neighbor table-B should ideally match chair-X's neighbor table-Y.
    """
    n1, n2 = len(feats_1), len(feats_2)
    n = cost_matrix.shape[0]

    # Build fast lookups: feat list index → node_idx, and reverse
    idx1_to_pos = {f.idx: i for i, f in enumerate(feats_1)}
    idx2_to_pos = {f.idx: j for j, f in enumerate(feats_2)}

    # Adjacency in terms of feature-list positions (not node_idx)
    adj1_pos: Dict[int, set] = {}
    for i, f in enumerate(feats_1):
        neighbors = adj_1.get(f.idx, [])
        adj1_pos[i] = {idx1_to_pos[n] for n in neighbors if n in idx1_to_pos}

    adj2_pos: Dict[int, set] = {}
    for j, f in enumerate(feats_2):
        neighbors = adj_2.get(f.idx, [])
        adj2_pos[j] = {idx2_to_pos[n] for n in neighbors if n in idx2_to_pos}

    working_cost = cost_matrix.copy()

    for iteration in range(n_iters):
        # Solve with current costs
        pairs = solve_assignment(working_cost, n1, n2, unmatched_cost)

        # Build match map: i → j (feature list positions)
        match_fwd = {}  # SG1 pos → SG2 pos
        match_rev = {}  # SG2 pos → SG1 pos
        for i, j, c in pairs:
            if c < unmatched_cost:
                match_fwd[i] = j
                match_rev[j] = i

        # Compute neighborhood consistency for each real pair
        adjustments = 0
        n_rewarded = 0
        total_consistency = 0.0
        n_scored = 0
        for i, j, c in pairs:
            if c >= unmatched_cost:
                continue
            neigh_i = adj1_pos.get(i, set())
            neigh_j = adj2_pos.get(j, set())
            if not neigh_i and not neigh_j:
                continue

            # Count how many neighbors of i are matched to neighbors of j
            consistent = 0
            for ni in neigh_i:
                matched_to = match_fwd.get(ni)
                if matched_to is not None and matched_to in neigh_j:
                    consistent += 1

            max_possible = max(len(neigh_i), len(neigh_j), 1)
            consistency_score = consistent / max_possible
            total_consistency += consistency_score
            n_scored += 1

            # Reward-only: only reduce cost for high-consistency pairs
            # Don't penalize low consistency (avoids error propagation)
            if consistency_score >= 0.2:
                delta = -bonus * consistency_score
                working_cost[i, j] = max(0.0, working_cost[i, j] + delta)
                n_rewarded += 1
            adjustments += 1

        avg_cons = total_consistency / max(n_scored, 1)
        print(f"[{_ts()}]   Refinement round {iteration+1}/{n_iters}: "
              f"{len(match_fwd)} matches, {n_rewarded} rewarded, "
              f"avg neighborhood consistency: {avg_cons:.3f}")

    # Final solve
    final_pairs = solve_assignment(working_cost, n1, n2, unmatched_cost)
    return final_pairs


# ---------------------------------------------------------------------------
# Edge consistency
# ---------------------------------------------------------------------------

def compute_edge_consistency(match_map: dict, edges_1: list,
                             edges_2: list) -> float:
    """Fraction of SG1 edges whose endpoints both map to SG2 and have a
    corresponding edge there."""
    if not edges_1:
        return 1.0

    edge_set_2 = set()
    for e in edges_2:
        pair = frozenset([e['node1_idx'], e['node2_idx']])
        edge_set_2.add(pair)

    consistent = 0
    total = 0
    for e in edges_1:
        mapped_n1 = match_map.get(e['node1_idx'])
        mapped_n2 = match_map.get(e['node2_idx'])
        if mapped_n1 is None or mapped_n2 is None:
            continue
        total += 1
        if frozenset([mapped_n1, mapped_n2]) in edge_set_2:
            consistent += 1

    return consistent / max(total, 1)


def compute_edge_relation_accuracy(match_map: dict, edges_1: list,
                                   edges_2: list) -> float:
    """Of edges that exist in both graphs, fraction with same relation text."""
    edge_dict_2 = {}
    for e in edges_2:
        pair = frozenset([e['node1_idx'], e['node2_idx']])
        edge_dict_2[pair] = e.get('relation', '')

    matched_relations = 0
    total = 0
    for e in edges_1:
        mapped_n1 = match_map.get(e['node1_idx'])
        mapped_n2 = match_map.get(e['node2_idx'])
        if mapped_n1 is None or mapped_n2 is None:
            continue
        pair = frozenset([mapped_n1, mapped_n2])
        if pair in edge_dict_2:
            total += 1
            rel1 = (e.get('relation') or '').lower().strip()
            rel2 = (edge_dict_2[pair] or '').lower().strip()
            if rel1 == rel2:
                matched_relations += 1

    return matched_relations / max(total, 1)


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------

@dataclass
class ChangeThresholds:
    """Thresholds for classifying matched pairs."""
    same_clip_min: float = 0.75
    same_dist_max: float = 2.0
    moved_clip_min: float = 0.70
    replaced_clip_max: float = 0.50
    replaced_dist_max: float = 1.5
    cost_reject: float = 0.42


def classify_match(fa: NodeFeatures, fb: NodeFeatures,
                   cost: float,
                   edge_consistency: float,
                   thresholds: ChangeThresholds = None) -> Tuple[ChangeType, float]:
    """Classify a matched pair into a ChangeType with confidence."""
    if thresholds is None:
        thresholds = ChangeThresholds()

    clip_sim = _cosine_sim(fa.clip_ft, fb.clip_ft)
    dist_3d = spatial_dist_meters(fa, fb)
    text_sim = _cosine_sim(fa.text_ft, fb.text_ft)
    label_sim = _label_similarity(fa.caption, fb.caption)

    if cost >= thresholds.cost_reject:
        return ChangeType.REMOVED, 0.5

    # Cross-category matches are assignment artifacts — reject them.
    # They "consume" slots in Hungarian but should not be reported as real matches.
    if label_sim == 0.0:
        return ChangeType.REMOVED, 0.5

    # SAME: high visual similarity + close spatial proximity
    if clip_sim >= thresholds.same_clip_min and dist_3d <= thresholds.same_dist_max:
        confidence = min(clip_sim, 1.0 - dist_3d / (thresholds.same_dist_max * 2))
        confidence = max(0.5, confidence)
        confidence = min(1.0, confidence + 0.1 * edge_consistency)
        return ChangeType.SAME, round(confidence, 3)

    # MOVED: high visual similarity but far apart
    # Require same or similar label to avoid cross-category false positives
    label_match = _label_similarity(fa.caption, fb.caption)
    if (clip_sim >= thresholds.moved_clip_min and
            dist_3d > thresholds.same_dist_max and
            label_match >= 0.5):
        confidence = clip_sim * 0.5 + text_sim * 0.2 + label_match * 0.3
        return ChangeType.MOVED, round(min(1.0, confidence), 3)

    # REPLACED: different object at same location
    if clip_sim <= thresholds.replaced_clip_max and dist_3d <= thresholds.replaced_dist_max:
        confidence = (1.0 - clip_sim) * 0.5 + (1.0 - dist_3d / thresholds.replaced_dist_max) * 0.5
        return ChangeType.REPLACED, round(min(1.0, confidence), 3)

    # UNCERTAIN
    return ChangeType.UNCERTAIN, round(max(0.1, 0.5 - abs(cost - 0.5)), 3)


# ---------------------------------------------------------------------------
# Main matching pipeline
# ---------------------------------------------------------------------------

def load_scene_graph(filepath: str) -> Tuple[dict, dict]:
    """Load a scene graph and maps from a .pkl file.

    Returns (scenegraph_dict, maps_dict).
    """
    print(f"[{_ts()}] Loading {filepath.split('/')[-1]} ...")
    with open(filepath, 'rb') as f:
        save_data = pickle.load(f)
    sg = save_data['scenegraph']
    maps = save_data.get('maps', {})
    print(f"[{_ts()}]   -> {len(sg['nodes'])} nodes, {len(sg['edges'])} edges")
    return sg, maps


def match_scene_graphs(sg1_path: str, sg2_path: str,
                       weights: MatchWeights = None,
                       thresholds: ChangeThresholds = None,
                       verbose: bool = False,
                       refine_iters: int = 0) -> MatchReport:
    """Full matching pipeline between two saved scene graphs."""
    if weights is None:
        weights = MatchWeights()
    if thresholds is None:
        thresholds = ChangeThresholds()

    sg1_data, maps1 = load_scene_graph(sg1_path)
    sg2_data, maps2 = load_scene_graph(sg2_path)

    feats_1 = extract_node_features(sg1_data, label="SG1")
    feats_2 = extract_node_features(sg2_data, label="SG2")

    snaps_1 = build_node_edge_snapshots(sg1_data, label="SG1")
    snaps_2 = build_node_edge_snapshots(sg2_data, label="SG2")

    # Build adjacency and compute neighborhood features (shared vocabulary)
    adj_1 = build_adjacency(sg1_data)
    adj_2 = build_adjacency(sg2_data)
    all_cats = sorted(set(
        f.caption.lower().strip().replace('_', ' ')
        for f in feats_1 + feats_2))
    cat_to_bin = {c: i for i, c in enumerate(all_cats)}
    compute_neighborhood_features(feats_1, adj_1, cat_to_bin, label="SG1")
    compute_neighborhood_features(feats_2, adj_2, cat_to_bin, label="SG2")

    # Auto-compute scene_scale if not overridden
    auto_scale = compute_auto_scene_scale(feats_1, feats_2)
    if weights.scene_scale <= 0:
        weights.scene_scale = auto_scale
    print(f"[{_ts()}] Scene scale: {weights.scene_scale:.1f}m "
          f"(auto={auto_scale:.1f}m)")

    # Build explored masks for UNSEEN detection
    print(f"[{_ts()}] Building exploration coverage masks ...")
    explored_mask_1 = build_explored_mask(maps1)
    explored_mask_2 = build_explored_mask(maps2)
    if explored_mask_1 is not None:
        pct1 = 100 * np.count_nonzero(explored_mask_1) / explored_mask_1.size
        print(f"[{_ts()}]   SG1 explored: {pct1:.1f}% of grid")
    if explored_mask_2 is not None:
        pct2 = 100 * np.count_nonzero(explored_mask_2) / explored_mask_2.size
        print(f"[{_ts()}]   SG2 explored: {pct2:.1f}% of grid")

    n1, n2 = len(feats_1), len(feats_2)

    if verbose:
        print(f"SG1: {n1} nodes, {len(sg1_data['edges'])} edges, "
              f"{len(snaps_1)} nodes with snapshots")
        print(f"SG2: {n2} nodes, {len(sg2_data['edges'])} edges, "
              f"{len(snaps_2)} nodes with snapshots")

    if n1 == 0 or n2 == 0:
        unmatched = []
        for f in feats_1:
            unmatched.append(asdict(UnmatchedNode(
                sg_source="sg1", node_idx=f.idx, caption=f.caption,
                change_type=ChangeType.REMOVED.value,
                centroid_3d=f.centroid_3d.tolist(), room_idx=f.room_idx)))
        for f in feats_2:
            unmatched.append(asdict(UnmatchedNode(
                sg_source="sg2", node_idx=f.idx, caption=f.caption,
                change_type=ChangeType.ADDED.value,
                centroid_3d=f.centroid_3d.tolist(), room_idx=f.room_idx)))
        return MatchReport(
            sg1_path=sg1_path, sg2_path=sg2_path,
            sg1_num_nodes=n1, sg2_num_nodes=n2,
            sg1_num_edges=len(sg1_data['edges']),
            sg2_num_edges=len(sg2_data['edges']),
            matches=[], unmatched=unmatched,
            edge_consistency_global=0.0,
            summary={t.value: 0 for t in ChangeType})

    # Build and solve cost matrix
    cost_matrix = build_cost_matrix(feats_1, feats_2, weights, snaps_1, snaps_2)

    if refine_iters > 0:
        print(f"[{_ts()}] Running iterative neighborhood consensus "
              f"({refine_iters} rounds) ...")
        pairs = iterative_neighborhood_consensus(
            cost_matrix, feats_1, feats_2, adj_1, adj_2,
            n_iters=refine_iters, unmatched_cost=weights.unmatched_cost)
    else:
        pairs = solve_assignment(cost_matrix, n1, n2, weights.unmatched_cost)

    print(f"[{_ts()}] Classifying {len(pairs)} matched pairs ...")

    if verbose:
        print(f"\n{'='*90}")
        print(f"{'SG1 Node':25s} {'SG2 Node':25s} {'Cost':>6s} {'CLIP':>6s} "
              f"{'Color':>5s} {'Dist':>7s} {'Type':>10s} {'Conf':>5s}")
        print(f"{'='*90}")

    # Classify each match
    matched_sg1 = set()
    matched_sg2 = set()
    matches = []

    for i, j, c in pairs:
        fa, fb = feats_1[i], feats_2[j]
        change_type, confidence = classify_match(
            fa, fb, c, 0.0, thresholds)

        if change_type == ChangeType.REMOVED:
            continue

        matched_sg1.add(i)
        matched_sg2.add(j)

        clip_sim = _cosine_sim(fa.clip_ft, fb.clip_ft)
        dist_3d = spatial_dist_meters(fa, fb)
        text_sim = _cosine_sim(fa.text_ft, fb.text_ft)
        chist_sim = 1.0 - color_hist_cost(fa, fb)

        matches.append(asdict(MatchResult(
            sg1_idx=fa.idx, sg2_idx=fb.idx,
            sg1_caption=fa.caption, sg2_caption=fb.caption,
            change_type=change_type.value,
            cost=round(c, 4),
            clip_visual_sim=round(clip_sim, 4),
            spatial_dist_3d=round(dist_3d, 4),
            clip_text_sim=round(text_sim, 4),
            color_hist_sim=round(chist_sim, 4),
            same_room=(fa.room_idx == fb.room_idx),
            edge_consistency=0.0,
            confidence=confidence,
        )))

        if verbose:
            print(f"  {fa.caption:23s} {fb.caption:23s} {c:6.3f} "
                  f"{clip_sim:6.3f} {chist_sim:5.3f} {dist_3d:6.2f}m "
                  f"{change_type.value:>10s} {confidence:5.3f}")

    # Unmatched nodes — distinguish UNSEEN (not explored) from REMOVED/ADDED
    print(f"[{_ts()}] Classifying unmatched nodes (REMOVED/UNSEEN/ADDED) ...")
    unmatched = []
    for i, f in enumerate(feats_1):
        if i not in matched_sg1:
            # Was this SG1 node's location explored by SG2?
            if is_in_explored_area(f.centroid_3d, explored_mask_2):
                ctype = ChangeType.REMOVED  # SG2 explored here but didn't see it
            else:
                ctype = ChangeType.UNSEEN   # SG2 never explored this area
            unmatched.append(asdict(UnmatchedNode(
                sg_source="sg1", node_idx=f.idx, caption=f.caption,
                change_type=ctype.value,
                centroid_3d=f.centroid_3d.tolist(), room_idx=f.room_idx)))

    for j, f in enumerate(feats_2):
        if j not in matched_sg2:
            # Was this SG2 node's location explored by SG1?
            if is_in_explored_area(f.centroid_3d, explored_mask_1):
                ctype = ChangeType.ADDED    # SG1 explored here but didn't see it
            else:
                ctype = ChangeType.UNSEEN   # SG1 never explored this area
            unmatched.append(asdict(UnmatchedNode(
                sg_source="sg2", node_idx=f.idx, caption=f.caption,
                change_type=ctype.value,
                centroid_3d=f.centroid_3d.tolist(), room_idx=f.room_idx)))

    # Edge consistency — computed on accepted matches only (post-classification)
    print(f"[{_ts()}] Computing edge consistency ({len(matches)} matches, "
          f"{len(sg1_data['edges'])}+{len(sg2_data['edges'])} edges) ...")
    match_map_node = {}
    for m in matches:
        match_map_node[m['sg1_idx']] = m['sg2_idx']
    edge_consistency = compute_edge_consistency(
        match_map_node, sg1_data['edges'], sg2_data['edges'])
    edge_relation_acc = compute_edge_relation_accuracy(
        match_map_node, sg1_data['edges'], sg2_data['edges'])

    # Backfill edge_consistency into match results
    for m in matches:
        m['edge_consistency'] = round(edge_consistency, 4)

    if verbose:
        print(f"\nEdge consistency (post-classification): {edge_consistency:.2%}")
        print(f"Edge relation accuracy: {edge_relation_acc:.2%}")

    # Summary
    summary = {t.value: 0 for t in ChangeType}
    for m in matches:
        summary[m['change_type']] += 1
    for u in unmatched:
        summary[u['change_type']] += 1

    report = MatchReport(
        sg1_path=sg1_path, sg2_path=sg2_path,
        sg1_num_nodes=n1, sg2_num_nodes=n2,
        sg1_num_edges=len(sg1_data['edges']),
        sg2_num_edges=len(sg2_data['edges']),
        matches=matches, unmatched=unmatched,
        edge_consistency_global=round(edge_consistency, 4),
        summary=summary,
    )

    if verbose:
        print(f"\n{'='*80}")
        if unmatched:
            print(f"\nUnmatched nodes:")
            for u in unmatched:
                print(f"  {u['caption']:23s} ({u['sg_source']}) -> {u['change_type']}")

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Match two SG-Nav scene graphs and detect changes")
    parser.add_argument('--sg1', required=True,
                        help='Path to first scene graph .pkl (reference)')
    parser.add_argument('--sg2', required=True,
                        help='Path to second scene graph .pkl (query)')
    parser.add_argument('--output', default=None,
                        help='Output JSON path (default: stdout summary)')
    parser.add_argument('--verbose', '-v', action='store_true')

    # Mode preset (sets default weights; individual --w-* flags override)
    parser.add_argument('--mode', choices=['static', 'dynamic'], default='static',
                        help='Matching mode: "static" (objects stay put) or '
                             '"dynamic" (objects may move/be removed)')

    # Weight overrides (default=None means "use preset")
    parser.add_argument('--w-spatial', type=float, default=None)
    parser.add_argument('--w-color', type=float, default=None)
    parser.add_argument('--w-nch', type=float, default=None)
    parser.add_argument('--w-clip-visual', type=float, default=None)
    parser.add_argument('--w-neighbor-clip', type=float, default=None)
    parser.add_argument('--w-snapshot', type=float, default=None)
    parser.add_argument('--w-label', type=float, default=None)
    parser.add_argument('--w-shape', type=float, default=None)
    parser.add_argument('--w-clip-text', type=float, default=None)
    parser.add_argument('--w-room', type=float, default=None)
    parser.add_argument('--unmatched-cost', type=float, default=None)
    parser.add_argument('--scene-scale', type=float, default=-1.0,
                        help='Scene scale in meters (-1 = auto from data)')

    # Iterative refinement
    parser.add_argument('--refine-iters', type=int, default=0,
                        help='Neighborhood consensus refinement iterations (0=off, 2-3 recommended)')

    # Change detection thresholds
    parser.add_argument('--same-clip-min', type=float, default=0.75)
    parser.add_argument('--same-dist-max', type=float, default=2.0)
    parser.add_argument('--moved-clip-min', type=float, default=0.70)

    args = parser.parse_args()

    # Start from preset, then apply any explicit overrides
    weights = MatchWeights.preset(args.mode)
    for attr, flag in [('w_spatial', 'w_spatial'), ('w_color', 'w_color'),
                       ('w_nch', 'w_nch'), ('w_clip_visual', 'w_clip_visual'),
                       ('w_neighbor_clip', 'w_neighbor_clip'),
                       ('w_snapshot', 'w_snapshot'),
                       ('w_label', 'w_label'), ('w_shape', 'w_shape'),
                       ('w_clip_text', 'w_clip_text'), ('w_room', 'w_room'),
                       ('unmatched_cost', 'unmatched_cost')]:
        val = getattr(args, flag)
        if val is not None:
            setattr(weights, attr, val)
    weights.scene_scale = args.scene_scale
    neigh_pct = weights.w_nch + weights.w_neighbor_clip
    pos_pct = weights.w_spatial + weights.w_room
    print(f"[{_ts()}] Mode: {args.mode} | Position: {pos_pct:.0%} | "
          f"Neighborhood: {neigh_pct:.0%} | "
          f"Identity: {1.0 - pos_pct - neigh_pct:.0%} | "
          f"Refine iters: {args.refine_iters}")

    # Mode-aware threshold defaults
    if args.mode == 'dynamic':
        default_same_dist = 5.0     # objects expected to move
        default_same_clip = 0.80    # rely more on identity
        default_moved_clip = 0.65   # accept looser identity for MOVED
    else:
        default_same_dist = 2.0
        default_same_clip = 0.75
        default_moved_clip = 0.70

    thresholds = ChangeThresholds(
        same_clip_min=args.same_clip_min if args.same_clip_min != 0.75 else default_same_clip,
        same_dist_max=args.same_dist_max if args.same_dist_max != 2.0 else default_same_dist,
        moved_clip_min=args.moved_clip_min if args.moved_clip_min != 0.70 else default_moved_clip,
        cost_reject=weights.unmatched_cost,
    )

    report = match_scene_graphs(
        args.sg1, args.sg2, weights, thresholds,
        verbose=args.verbose, refine_iters=args.refine_iters)

    # ---- Print structured summary ----
    n1, n2 = report.sg1_num_nodes, report.sg2_num_nodes
    n_matched = sum(1 for m in report.matches
                    if m['change_type'] in ('SAME', 'MOVED', 'REPLACED'))
    n_uncertain = sum(1 for m in report.matches
                      if m['change_type'] == 'UNCERTAIN')
    max_possible = min(n1, n2)
    match_pct = 100 * n_matched / max_possible if max_possible else 0

    # Unmatched breakdowns by source
    unmatched_sg1 = [u for u in report.unmatched if u['sg_source'] == 'sg1']
    unmatched_sg2 = [u for u in report.unmatched if u['sg_source'] == 'sg2']
    unseen_sg1 = [u for u in unmatched_sg1 if u['change_type'] == 'UNSEEN']
    unseen_sg2 = [u for u in unmatched_sg2 if u['change_type'] == 'UNSEEN']
    removed = [u for u in unmatched_sg1 if u['change_type'] == 'REMOVED']
    added = [u for u in unmatched_sg2 if u['change_type'] == 'ADDED']

    # Pre-compute stats for SAME and MOVED
    same_matches = [m for m in report.matches if m['change_type'] == 'SAME']
    moved_matches = [m for m in report.matches if m['change_type'] == 'MOVED']

    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    same_clips = [m['clip_visual_sim'] for m in same_matches]
    same_colors = [m['color_hist_sim'] for m in same_matches]
    same_dists = [m['spatial_dist_3d'] for m in same_matches]
    same_confs = [m['confidence'] for m in same_matches]
    moved_dists = [m['spatial_dist_3d'] for m in moved_matches]
    moved_clips = [m['clip_visual_sim'] for m in moved_matches]
    moved_colors = [m['color_hist_sim'] for m in moved_matches]

    # Size asymmetry breakdown
    if n1 > n2:
        n_capacity_overflow = n1 - n2
        n_rejected = max(0, len(unmatched_sg1) - len(unseen_sg1) - n_capacity_overflow)
    else:
        n_capacity_overflow = 0
        n_rejected = len(removed)
    if n2 > n1:
        n_capacity_overflow_sg2 = n2 - n1
        n_rejected_sg2 = max(0, len(unmatched_sg2) - len(unseen_sg2) - n_capacity_overflow_sg2)
    else:
        n_capacity_overflow_sg2 = 0
        n_rejected_sg2 = len(added)

    # ================================================================
    # RESULTS TABLE
    # ================================================================
    W = 66
    sep = '=' * W
    thin = '-' * W

    print(f"\n{sep}")
    print(f"  MATCHING RESULTS  (mode={args.mode})")
    print(f"{sep}")

    # ---- Overview table ----
    print(f"\n  {'Metric':<36s} {'Value':>26s}")
    print(f"  {thin}")
    print(f"  {'SG1 nodes / edges':<36s} {n1:>12d} / {report.sg1_num_edges:<12d}")
    print(f"  {'SG2 nodes / edges':<36s} {n2:>12d} / {report.sg2_num_edges:<12d}")
    print(f"  {'Edge consistency':<36s} {report.edge_consistency_global:>25.1%}")
    print(f"  {'Match rate (of smaller graph)':<36s} {n_matched:>5d} / {max_possible} ({match_pct:.1f}%)")

    # ---- Match classification table ----
    print(f"\n  {'Change Type':<16s} {'Count':>7s} {'Avg Dist':>10s} "
          f"{'Avg CLIP':>10s} {'Avg Color':>10s}")
    print(f"  {thin}")
    if same_matches:
        print(f"  {'SAME':<16s} {len(same_matches):>7d} {_avg(same_dists):>9.2f}m "
              f"{_avg(same_clips):>10.3f} {_avg(same_colors):>10.3f}")
    if moved_matches:
        print(f"  {'MOVED':<16s} {len(moved_matches):>7d} {_avg(moved_dists):>9.2f}m "
              f"{_avg(moved_clips):>10.3f} {_avg(moved_colors):>10.3f}")
    replaced = [m for m in report.matches if m['change_type'] == 'REPLACED']
    if replaced:
        print(f"  {'REPLACED':<16s} {len(replaced):>7d}")
    if n_uncertain:
        print(f"  {'UNCERTAIN':<16s} {n_uncertain:>7d}")
    print(f"  {thin}")
    print(f"  {'Total matched':<16s} {n_matched + n_uncertain:>7d}")

    # ---- Unmatched table ----
    print(f"\n  {'Unmatched Nodes':<36s} {'SG1':>8s} {'SG2':>8s}")
    print(f"  {thin}")
    if n_capacity_overflow > 0:
        print(f"  {'Size asymmetry (no counterpart)':<36s} {n_capacity_overflow:>8d} {'—':>8s}")
    if n_capacity_overflow_sg2 > 0:
        print(f"  {'Size asymmetry (no counterpart)':<36s} {'—':>8s} {n_capacity_overflow_sg2:>8d}")
    print(f"  {'Rejected (cost/label mismatch)':<36s} "
          f"{n_rejected:>8d} {n_rejected_sg2:>8d}")
    print(f"  {'UNSEEN (unexplored area)':<36s} "
          f"{len(unseen_sg1):>8d} {len(unseen_sg2):>8d}")
    print(f"  {thin}")
    print(f"  {'Total unmatched':<36s} "
          f"{len(unmatched_sg1):>8d} {len(unmatched_sg2):>8d}")

    # ---- SAME quality details ----
    if same_matches:
        print(f"\n  SAME Quality (n={len(same_matches)}):")
        print(f"  {'Metric':<28s} {'Avg':>8s} {'Min':>8s} {'Max':>8s}")
        print(f"  {thin}")
        print(f"  {'CLIP visual sim':<28s} {_avg(same_clips):>8.3f} "
              f"{min(same_clips):>8.3f} {max(same_clips):>8.3f}")
        print(f"  {'Color histogram sim':<28s} {_avg(same_colors):>8.3f} "
              f"{min(same_colors):>8.3f} {max(same_colors):>8.3f}")
        print(f"  {'Spatial distance (m)':<28s} {_avg(same_dists):>8.2f} "
              f"{min(same_dists):>8.2f} {max(same_dists):>8.2f}")
        print(f"  {'Confidence':<28s} {_avg(same_confs):>8.3f} "
              f"{min(same_confs):>8.3f} {max(same_confs):>8.3f}")

        from collections import Counter
        cat_counts = Counter(m['sg1_caption'] for m in same_matches)
        top_cats = cat_counts.most_common(8)
        cats_str = ', '.join(f"{cat}({n})" for cat, n in top_cats)
        print(f"  Top categories: {cats_str}")

    # ---- MOVED details ----
    if moved_matches:
        print(f"\n  MOVED Details (n={len(moved_matches)}):")
        print(f"  {'Metric':<28s} {'Avg':>8s} {'Min':>8s} {'Max':>8s}")
        print(f"  {thin}")
        print(f"  {'Spatial distance (m)':<28s} {_avg(moved_dists):>8.2f} "
              f"{min(moved_dists):>8.2f} {max(moved_dists):>8.2f}")
        print(f"  {'CLIP visual sim':<28s} {_avg(moved_clips):>8.3f} "
              f"{min(moved_clips):>8.3f} {max(moved_clips):>8.3f}")
        print(f"  {'Color histogram sim':<28s} {_avg(moved_colors):>8.3f} "
              f"{min(moved_colors):>8.3f} {max(moved_colors):>8.3f}")

    # ---- Config footer ----
    neigh_pct = weights.w_nch + weights.w_neighbor_clip
    pos_pct = weights.w_spatial + weights.w_room
    id_pct = 1.0 - pos_pct - neigh_pct
    print(f"\n  Config:")
    print(f"  {'Weight split':<28s}  Position {pos_pct:.0%} | "
          f"Neighborhood {neigh_pct:.0%} | Identity {id_pct:.0%}")
    print(f"  {'Thresholds':<28s}  same_dist≤{thresholds.same_dist_max}m  "
          f"same_clip≥{thresholds.same_clip_min}  "
          f"moved_clip≥{thresholds.moved_clip_min}")
    if args.refine_iters > 0:
        print(f"  {'Refinement':<28s}  {args.refine_iters} iterations")

    print(f"{sep}")

    if args.output:
        result = asdict(report)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nFull report saved to {args.output}")


if __name__ == '__main__':
    main()
