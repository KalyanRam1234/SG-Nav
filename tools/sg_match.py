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
    room_caption: Optional[str]      # room type name (e.g. "kitchen")
    num_detections: int
    num_points: int                  # total points in point cloud
    bbox_volume: float               # oriented bounding box volume
    color_hist: Optional[np.ndarray] = None   # (48,) HSV histogram
    shape_desc: Optional[np.ndarray] = None   # (4,) PCA eigenratios + log-volume
    neighbor_cat_hist: Optional[np.ndarray] = None  # (C,) neighbor category histogram
    neighbor_clip_agg: Optional[np.ndarray] = None  # (512,) mean neighbor CLIP
    relation_hist: Optional[np.ndarray] = None       # (R,) spatial relation type histogram


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
    sg1_room: Optional[str]
    sg2_room: Optional[str]
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
    room_caption: Optional[str]


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
    edge_consistency_soft: float
    edge_consistency_detail: dict       # EdgeConsistencyReport as dict
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

    # Build room_idx → caption lookup from room_nodes
    room_captions: Dict[int, str] = {}
    for ri, rn in enumerate(sg_data.get('room_nodes', [])):
        room_captions[ri] = rn.get('caption', '').lower().strip()

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

        ri = s_node.get('room_idx')
        features.append(NodeFeatures(
            idx=idx,
            caption=s_node.get('caption', ''),
            clip_ft=clip_ft,
            text_ft=text_ft,
            centroid_3d=centroid_3d,
            center_2d=s_node.get('center'),
            room_idx=ri,
            room_caption=room_captions.get(ri) if ri is not None else None,
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
    """Build a map: node_idx -> (K, 512) matrix of L2-normalized edge snapshot CLIPs.

    Legacy function kept for reference; see build_rich_snapshots() for the
    neighbor-keyed version with bbox geometry.
    """
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


@dataclass
class RichSnapshot:
    """Per-edge snapshot data enriched with bbox geometry."""
    neighbor_idx: int
    neighbor_caption: str
    clip_feature: np.ndarray     # (512,) L2-normalized CLIP of full scene image
    rel_offset: np.ndarray       # (2,) normalized offset (dx/diag, dy/diag) to neighbor
    log_size_ratio: float        # log(my_area / neighbor_area)
    my_log_aspect: float         # log(width / height) of my bbox


def _extract_bbox_geometry(my_bbox: list, neighbor_bbox: list
                           ) -> Tuple[np.ndarray, float, float]:
    """Extract relative geometry from two bounding boxes.

    Returns (rel_offset, log_size_ratio, my_log_aspect).
    rel_offset is the (dx, dy) from my center to neighbor center,
    normalized by the image diagonal so values are scale-invariant.
    """
    x1a, y1a, x2a, y2a = my_bbox
    x1b, y1b, x2b, y2b = neighbor_bbox
    my_cx, my_cy = (x1a + x2a) / 2, (y1a + y2a) / 2
    nb_cx, nb_cy = (x1b + x2b) / 2, (y1b + y2b) / 2

    diag = max(np.sqrt(640**2 + 480**2), 1.0)
    rel_offset = np.array([(nb_cx - my_cx) / diag,
                           (nb_cy - my_cy) / diag], dtype=np.float32)

    my_w, my_h = max(x2a - x1a, 1.0), max(y2a - y1a, 1.0)
    nb_w, nb_h = max(x2b - x1b, 1.0), max(y2b - y1b, 1.0)
    my_area = my_w * my_h
    nb_area = nb_w * nb_h
    log_size_ratio = float(np.log(my_area / nb_area))
    my_log_aspect = float(np.log(my_w / my_h))

    return rel_offset, log_size_ratio, my_log_aspect


def build_rich_snapshots(sg_data: dict, label: str = "SG"
                         ) -> Dict[int, List[RichSnapshot]]:
    """Build neighbor-keyed rich snapshots: node_idx -> [RichSnapshot, ...].

    Each snapshot captures the CLIP feature of the scene image showing
    this node with a specific neighbor, plus bbox geometric features
    describing their relative layout in the image.
    """
    print(f"[{_ts()}] Building rich edge snapshots for {label} ...")
    captions: Dict[int, str] = {}
    for idx, sn in enumerate(sg_data['nodes']):
        captions[idx] = (sn.get('caption') or '').lower().strip().replace('_', ' ')

    result: Dict[int, List[RichSnapshot]] = defaultdict(list)
    n_with_geo = 0
    n_clip_only = 0

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

        n1, n2 = e['node1_idx'], e['node2_idx']
        bboxes = e.get('snapshot_bboxes') or {}

        has_geo = ('node1' in bboxes and 'node2' in bboxes and
                   len(bboxes.get('node1', [])) == 4 and
                   len(bboxes.get('node2', [])) == 4)

        if has_geo:
            n_with_geo += 1
            b1, b2 = bboxes['node1'], bboxes['node2']
            off1, lsr1, la1 = _extract_bbox_geometry(b1, b2)
            result[n1].append(RichSnapshot(
                neighbor_idx=n2, neighbor_caption=captions.get(n2, ''),
                clip_feature=feat_norm,
                rel_offset=off1, log_size_ratio=lsr1, my_log_aspect=la1))
            off2, lsr2, la2 = _extract_bbox_geometry(b2, b1)
            result[n2].append(RichSnapshot(
                neighbor_idx=n1, neighbor_caption=captions.get(n1, ''),
                clip_feature=feat_norm,
                rel_offset=off2, log_size_ratio=lsr2, my_log_aspect=la2))
        else:
            n_clip_only += 1
            zero_off = np.zeros(2, dtype=np.float32)
            result[n1].append(RichSnapshot(
                neighbor_idx=n2, neighbor_caption=captions.get(n2, ''),
                clip_feature=feat_norm,
                rel_offset=zero_off, log_size_ratio=0.0, my_log_aspect=0.0))
            result[n2].append(RichSnapshot(
                neighbor_idx=n1, neighbor_caption=captions.get(n1, ''),
                clip_feature=feat_norm,
                rel_offset=zero_off, log_size_ratio=0.0, my_log_aspect=0.0))

    print(f"[{_ts()}]   -> {len(result)} nodes with snapshots "
          f"({n_with_geo} edges with bbox geometry, {n_clip_only} CLIP-only)")
    return dict(result)


def _snapshot_geo_similarity(a: RichSnapshot, b: RichSnapshot) -> float:
    """Geometric similarity between two rich snapshots.

    Compares relative offset, size ratio, and aspect ratio using
    Gaussian kernels. Returns value in [0, 1].
    """
    offset_diff = float(np.linalg.norm(a.rel_offset - b.rel_offset))
    offset_sim = float(np.exp(-offset_diff**2 / (2 * 0.15**2)))

    size_diff = abs(a.log_size_ratio - b.log_size_ratio)
    size_sim = float(np.exp(-size_diff**2 / (2 * 0.8**2)))

    aspect_diff = abs(a.my_log_aspect - b.my_log_aspect)
    aspect_sim = float(np.exp(-aspect_diff**2 / (2 * 0.5**2)))

    return 0.50 * offset_sim + 0.25 * size_sim + 0.25 * aspect_sim


def rich_snapshot_cost(snaps_a: Optional[List[RichSnapshot]],
                       snaps_b: Optional[List[RichSnapshot]]) -> float:
    """Neighbor-keyed snapshot matching cost.

    Groups snapshots by neighbor category and only compares snapshots
    sharing the same neighbor type. Within each shared category, combines
    CLIP similarity with bbox geometric similarity. Falls back to best
    overall CLIP match if no neighbor categories overlap.
    """
    if not snaps_a or not snaps_b:
        return 0.5

    groups_a: Dict[str, List[RichSnapshot]] = defaultdict(list)
    groups_b: Dict[str, List[RichSnapshot]] = defaultdict(list)
    for s in snaps_a:
        groups_a[s.neighbor_caption].append(s)
    for s in snaps_b:
        groups_b[s.neighbor_caption].append(s)

    shared_cats = set(groups_a.keys()) & set(groups_b.keys())

    if shared_cats:
        cat_scores = []
        for cat in shared_cats:
            best_score = 0.0
            for sa in groups_a[cat]:
                for sb in groups_b[cat]:
                    clip_sim = float(np.dot(sa.clip_feature, sb.clip_feature))
                    geo_sim = _snapshot_geo_similarity(sa, sb)
                    combined = 0.65 * clip_sim + 0.35 * geo_sim
                    best_score = max(best_score, combined)
            cat_scores.append(best_score)
        # Use max across categories (most discriminative shared snapshot)
        # rather than mean (which dilutes signal from many categories)
        return 1.0 - float(max(cat_scores))

    # Fallback: no shared categories → best overall CLIP match
    best_clip = 0.0
    for sa in snaps_a:
        for sb in snaps_b:
            sim = float(np.dot(sa.clip_feature, sb.clip_feature))
            best_clip = max(best_clip, sim)
    return 1.0 - best_clip


def build_adjacency(sg_data: dict) -> Dict[int, List[int]]:
    """Build adjacency lists: node_idx -> sorted list of neighbor node_idx."""
    adj: Dict[int, set] = defaultdict(set)
    for e in sg_data.get('edges', []):
        n1, n2 = e['node1_idx'], e['node2_idx']
        adj[n1].add(n2)
        adj[n2].add(n1)
    return {k: sorted(v) for k, v in adj.items()}


# Canonical spatial relation types extracted from edge relation text.
# Order matters: earlier patterns take precedence for ambiguous phrases.
RELATION_TYPES = [
    'on top', 'inside', 'next to', 'under', 'above',
    'below', 'left', 'right', 'front', 'behind', 'near',
]
N_RELATION_TYPES = len(RELATION_TYPES) + 1  # +1 for "other/unknown"
_RELATION_KEYWORDS = [(rt, rt) for rt in RELATION_TYPES]


def parse_spatial_relation(relation_text: str) -> int:
    """Extract canonical spatial relation type index from LLM-generated text.

    Returns index into RELATION_TYPES, or len(RELATION_TYPES) for unknown.
    The relation text is like "the rug is located in front of the door".
    """
    text = (relation_text or '').lower()
    for i, rt in enumerate(RELATION_TYPES):
        if rt in text:
            return i
    return len(RELATION_TYPES)  # "other"


def build_edge_relations(sg_data: dict) -> Dict[int, List[Tuple[int, int]]]:
    """Build per-node edge relation data: node_idx -> [(neighbor_idx, relation_type_idx), ...].

    For each edge, both endpoints get the relation. node1 gets the relation
    as-is; node2 gets the same (relations are symmetric for our histogram).
    """
    rels: Dict[int, list] = defaultdict(list)
    for e in sg_data.get('edges', []):
        n1, n2 = e['node1_idx'], e['node2_idx']
        rel_idx = parse_spatial_relation(e.get('relation', ''))
        rels[n1].append((n2, rel_idx))
        rels[n2].append((n1, rel_idx))
    return dict(rels)


def compute_neighborhood_features(feats: List[NodeFeatures],
                                  adj: Dict[int, List[int]],
                                  edge_rels: Dict[int, List[Tuple[int, int]]],
                                  cat_to_bin: Dict[str, int],
                                  label: str = "SG") -> None:
    """Fill in neighbor_cat_hist, neighbor_clip_agg, and relation_hist
    for each node in-place.

    neighbor_cat_hist: normalized histogram over neighbor categories
    neighbor_clip_agg: mean L2-normalized CLIP embedding of neighbors
    relation_hist: histogram over spatial relation types (left/right/above/...)

    cat_to_bin must be a shared vocabulary across both graphs so that
    histograms are directly comparable.
    """
    print(f"[{_ts()}] Computing neighborhood features for {label} ...")
    idx_to_feat: Dict[int, NodeFeatures] = {f.idx: f for f in feats}
    n_cats = len(cat_to_bin)
    n_rels = N_RELATION_TYPES

    n_with_neighbors = 0
    n_with_relations = 0
    for f in feats:
        neighbors = adj.get(f.idx, [])
        neighbor_feats = [idx_to_feat[n] for n in neighbors if n in idx_to_feat]

        if not neighbor_feats:
            f.neighbor_cat_hist = np.zeros(n_cats, dtype=np.float32)
            f.neighbor_clip_agg = np.zeros_like(f.clip_ft)
            f.relation_hist = np.zeros(n_rels, dtype=np.float32)
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

        # Relation type histogram
        rel_hist = np.zeros(n_rels, dtype=np.float32)
        node_rels = edge_rels.get(f.idx, [])
        if node_rels:
            n_with_relations += 1
            for _, rel_type_idx in node_rels:
                rel_hist[rel_type_idx] += 1.0
            rel_total = rel_hist.sum()
            if rel_total > 0:
                rel_hist /= rel_total

        f.relation_hist = rel_hist

    print(f"[{_ts()}]   -> {n_with_neighbors}/{len(feats)} nodes with ≥1 neighbor, "
          f"{n_with_relations} with relations, "
          f"{n_cats} cat bins, {n_rels} relation types")


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------

# Semantic room similarity (symmetric pairs, sorted alphabetically).
# Values are similarity in [0,1]: 1.0 = identical, 0.0 = unrelated.
# Only non-zero off-diagonal pairs need entries; same-caption is always 1.0.
ROOM_SEMANTIC_SIM: Dict[Tuple[str, str], float] = {
    ('dining room', 'kitchen'):     0.65,
    ('dining room', 'living room'): 0.45,
    ('kitchen', 'living room'):     0.35,
    ('kitchen', 'laundry room'):    0.35,
    ('bedroom', 'living room'):     0.25,
    ('bedroom', 'office room'):     0.20,
    ('bathroom', 'laundry room'):   0.30,
    ('bathroom', 'bedroom'):        0.15,
    ('living room', 'lounge'):      0.60,
    ('bedroom', 'lounge'):          0.30,
    ('gym', 'lounge'):              0.20,
    ('living room', 'office room'): 0.25,
    ('dining room', 'lounge'):      0.35,
}


def build_room_sim_table() -> Dict[Tuple[str, str], float]:
    """Return the room similarity lookup (keys sorted alphabetically)."""
    return dict(ROOM_SEMANTIC_SIM)


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


def room_cost(a: NodeFeatures, b: NodeFeatures,
              room_sim: Optional[Dict[Tuple[str, str], float]] = None) -> float:
    """Room similarity cost with soft semantic matching.

    Uses room captions (not indices) so the comparison is robust across graphs
    with different indexing.  Semantically related rooms (kitchen/dining) get
    partial similarity via room_sim lookup.
    """
    rc_a, rc_b = a.room_caption, b.room_caption
    if rc_a is None or rc_b is None:
        return 0.5
    if rc_a == rc_b:
        return 0.0
    # Look up soft semantic similarity if provided
    if room_sim is not None:
        key = (rc_a, rc_b) if rc_a <= rc_b else (rc_b, rc_a)
        if key in room_sim:
            return 1.0 - room_sim[key]   # similarity → cost
    return 1.0


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


def relation_hist_cost(a: NodeFeatures, b: NodeFeatures) -> float:
    """Cost from spatial relation type histogram intersection.

    Compares the distribution of edge relation types (left, above, behind, ...)
    around each node. A node that is mostly "above" and "left of" its neighbors
    should match another node with similar relation distribution.
    """
    if a.relation_hist is None or b.relation_hist is None:
        return 0.5
    if a.relation_hist.sum() < 1e-8 or b.relation_hist.sum() < 1e-8:
        return 0.5
    intersection = np.minimum(a.relation_hist, b.relation_hist).sum()
    return 1.0 - float(intersection)


# ---------------------------------------------------------------------------
# Cost matrix & Hungarian assignment
# ---------------------------------------------------------------------------

@dataclass
class MatchWeights:
    """Weights for combining cost components (11 components).

    Two presets available via `MatchWeights.preset()`:
      - "static"  — spatial position is primary signal (objects don't move)
      - "dynamic" — identity + neighborhood is primary signal (objects may move)
    """
    w_spatial: float = 0.20
    w_color: float = 0.15
    w_nch: float = 0.10
    w_clip_visual: float = 0.10
    w_neighbor_clip: float = 0.06
    w_snapshot: float = 0.08
    w_label: float = 0.06
    w_shape: float = 0.04
    w_clip_text: float = 0.03
    w_room: float = 0.10
    w_rel_hist: float = 0.08
    unmatched_cost: float = 0.42
    scene_scale: float = 5.0
    max_match_dist: float = -1.0   # hard spatial gate (m); <0 = auto

    @staticmethod
    def preset(mode: str) -> 'MatchWeights':
        if mode == "static":
            # Position: 30%, Neighborhood+Relations: 22%, Identity: 48%
            return MatchWeights(
                w_spatial=0.18, w_color=0.14, w_nch=0.09,
                w_clip_visual=0.10, w_neighbor_clip=0.06,
                w_snapshot=0.11, w_label=0.06, w_shape=0.05,
                w_clip_text=0.03, w_room=0.12,
                w_rel_hist=0.06,
                unmatched_cost=0.42,
                max_match_dist=5.0)
        elif mode == "dynamic":
            # Position: 18%, Neighborhood+Relations: 28%, Identity: 54%
            # Spatial boosted from 4%→10% to prevent wrong-instance matches
            # among dense same-category clusters (e.g. 154 windows in bedroom)
            return MatchWeights(
                w_spatial=0.10, w_color=0.18, w_nch=0.10,
                w_clip_visual=0.13, w_neighbor_clip=0.07,
                w_snapshot=0.10, w_label=0.06, w_shape=0.04,
                w_clip_text=0.04, w_room=0.08,
                w_rel_hist=0.10,
                unmatched_cost=0.38,
                max_match_dist=8.0)
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'static' or 'dynamic'.")


def compute_auto_max_match_dist(feats_1: list, feats_2: list,
                                percentile: float = 90) -> float:
    """Compute a reasonable max_match_dist from nearest-neighbor statistics.

    For each SG2 node, finds the nearest same-category SG1 node.
    Returns the given percentile of those distances (default P90).
    Falls back to scene_scale/2 if too few same-category pairs exist.
    """
    nn_dists = []
    cat_to_sg1 = defaultdict(list)
    for f in feats_1:
        cat_to_sg1[f.caption.lower().strip().replace('_', ' ')].append(f)

    for f2 in feats_2:
        cat = f2.caption.lower().strip().replace('_', ' ')
        same_cat = cat_to_sg1.get(cat, [])
        if not same_cat:
            continue
        min_dist = min(float(np.linalg.norm(f2.centroid_3d - f1.centroid_3d))
                       for f1 in same_cat)
        nn_dists.append(min_dist)

    if len(nn_dists) < 10:
        return 10.0

    p_val = float(np.percentile(nn_dists, percentile))
    # Add 50% margin for SLAM drift, clamp to [3, 15]
    result = max(3.0, min(15.0, p_val * 1.5))
    return result


def build_cost_matrix(feats_1: list, feats_2: list,
                      weights: MatchWeights = None,
                      snaps_1: Dict[int, List[RichSnapshot]] = None,
                      snaps_2: Dict[int, List[RichSnapshot]] = None,
                      room_sim: Dict[Tuple[str, str], float] = None) -> np.ndarray:
    """Build the padded cost matrix for Hungarian assignment.

    Returns an (n1+n2) x (n1+n2) matrix where dummy entries have
    cost = weights.unmatched_cost.  Applies hard spatial gating:
    pairs beyond max_match_dist are set to unmatched_cost, preventing
    Hungarian from creating distant false matches among same-category
    duplicates.
    """
    if weights is None:
        weights = MatchWeights()

    n1, n2 = len(feats_1), len(feats_2)
    n = n1 + n2
    max_dist = weights.max_match_dist

    cost = np.full((n, n), weights.unmatched_cost, dtype=np.float64)

    total_cells = n1 * n2
    report_interval = max(1, n1 // 10)
    t0 = time.time()
    n_gated = 0
    print(f"[{_ts()}] Building cost matrix ({n1} x {n2} = {total_cells:,} cells, "
          f"spatial gate={max_dist:.1f}m) ...")

    for i, fa in enumerate(feats_1):
        if report_interval >= 5 and i % report_interval == 0 and i > 0:
            elapsed = time.time() - t0
            pct = 100 * i / n1
            eta = elapsed / i * (n1 - i)
            print(f"[{_ts()}]   {pct:5.1f}% ({i}/{n1} rows, "
                  f"{elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)")
        sa = snaps_1.get(fa.idx) if snaps_1 else None
        for j, fb in enumerate(feats_2):
            # Hard spatial gating: skip distant pairs entirely
            raw_dist = float(np.linalg.norm(fa.centroid_3d - fb.centroid_3d))
            if max_dist > 0 and raw_dist > max_dist:
                n_gated += 1
                continue  # cost stays at unmatched_cost

            sb = snaps_2.get(fb.idx) if snaps_2 else None
            c_vis = clip_visual_cost(fa, fb)
            c_spa = spatial_cost(fa, fb, weights.scene_scale)
            c_lbl = label_cost(fa, fb)
            c_txt = clip_text_cost(fa, fb)
            c_rm = room_cost(fa, fb, room_sim)
            c_snap = rich_snapshot_cost(sa, sb)
            c_color = color_hist_cost(fa, fb)
            c_shape = shape_cost(fa, fb)
            c_nch = neighbor_category_cost(fa, fb)
            c_nclip = neighbor_clip_cost(fa, fb)
            c_rh = relation_hist_cost(fa, fb)

            cost[i, j] = (weights.w_clip_visual * c_vis +
                          weights.w_spatial * c_spa +
                          weights.w_label * c_lbl +
                          weights.w_clip_text * c_txt +
                          weights.w_room * c_rm +
                          weights.w_snapshot * c_snap +
                          weights.w_color * c_color +
                          weights.w_shape * c_shape +
                          weights.w_nch * c_nch +
                          weights.w_neighbor_clip * c_nclip +
                          weights.w_rel_hist * c_rh)

    elapsed = time.time() - t0
    gated_pct = 100 * n_gated / total_cells if total_cells else 0
    print(f"[{_ts()}]   Cost matrix built in {elapsed:.1f}s "
          f"({n_gated:,} cells gated = {gated_pct:.1f}%)")
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
        proximity_m: float = 3.0,
        unmatched_cost: float = 0.42) -> list:
    """Iterative refinement: adjust costs based on soft neighborhood consistency.

    Uses proximity-based consistency to handle co-visibility differences:
    instead of requiring an exact edge in the target graph, checks whether
    matched neighbors are spatially close (within proximity_m).

    For each matched pair (a→b), computes what fraction of a's matched
    neighbors land close to b in 3D space. High consistency → reduce cost;
    low → increase cost. Re-solve and repeat.
    """
    n1, n2 = len(feats_1), len(feats_2)

    # Build fast lookups
    idx1_to_pos = {f.idx: i for i, f in enumerate(feats_1)}
    idx2_to_pos = {f.idx: j for j, f in enumerate(feats_2)}

    # Adjacency in terms of feature-list positions
    adj1_pos: Dict[int, set] = {}
    for i, f in enumerate(feats_1):
        neighbors = adj_1.get(f.idx, [])
        adj1_pos[i] = {idx1_to_pos[n] for n in neighbors if n in idx1_to_pos}

    adj2_pos: Dict[int, set] = {}
    for j, f in enumerate(feats_2):
        neighbors = adj_2.get(f.idx, [])
        adj2_pos[j] = {idx2_to_pos[n] for n in neighbors if n in idx2_to_pos}

    # SG2 edge set for hard consistency check
    # (not used for score calculation, but tracked for logging)
    edge_set_2_pos = set()
    for j_src, neigh_set in adj2_pos.items():
        for j_dst in neigh_set:
            edge_set_2_pos.add(frozenset([j_src, j_dst]))

    # 3D positions for proximity check
    pos2 = {j: feats_2[j].centroid_3d for j in range(n2)}

    working_cost = cost_matrix.copy()

    for iteration in range(n_iters):
        pairs = solve_assignment(working_cost, n1, n2, unmatched_cost)

        match_fwd = {}  # SG1 pos → SG2 pos
        for i, j, c in pairs:
            if c < unmatched_cost:
                match_fwd[i] = j

        n_rewarded = 0
        n_penalized = 0
        total_soft_cons = 0.0
        total_hard_cons = 0.0
        n_scored = 0
        for i, j, c in pairs:
            if c >= unmatched_cost:
                continue
            neigh_i = adj1_pos.get(i, set())
            if not neigh_i:
                continue

            # For each neighbor of i that is matched, check proximity to j
            n_matched_neighbors = 0
            n_close = 0
            n_exact = 0
            for ni in neigh_i:
                mj = match_fwd.get(ni)
                if mj is None:
                    continue
                n_matched_neighbors += 1
                # Hard: exact edge in SG2
                if frozenset([j, mj]) in edge_set_2_pos:
                    n_exact += 1
                    n_close += 1
                # Soft: proximity-based
                elif mj in pos2 and j in pos2:
                    d = float(np.linalg.norm(pos2[mj] - pos2[j]))
                    if d < proximity_m:
                        n_close += 1

            if n_matched_neighbors == 0:
                continue

            soft_score = n_close / n_matched_neighbors
            hard_score = n_exact / n_matched_neighbors
            total_soft_cons += soft_score
            total_hard_cons += hard_score
            n_scored += 1

            # Reward high soft consistency
            if soft_score >= 0.3:
                delta = -bonus * soft_score
                working_cost[i, j] = max(0.0, working_cost[i, j] + delta)
                n_rewarded += 1
            # Penalize very low consistency (no close neighbors at all)
            elif soft_score < 0.1 and n_matched_neighbors >= 2:
                delta = penalty * (1.0 - soft_score)
                working_cost[i, j] = min(unmatched_cost, working_cost[i, j] + delta)
                n_penalized += 1

        avg_soft = total_soft_cons / max(n_scored, 1)
        avg_hard = total_hard_cons / max(n_scored, 1)
        print(f"[{_ts()}]   Refinement round {iteration+1}/{n_iters}: "
              f"{len(match_fwd)} matches, {n_rewarded} rewarded, "
              f"{n_penalized} penalized, "
              f"avg consistency: hard={avg_hard:.3f} soft={avg_soft:.3f}")

    final_pairs = solve_assignment(working_cost, n1, n2, unmatched_cost)
    return final_pairs


def spatial_reassignment(pairs: list,
                         feats_1: List[NodeFeatures],
                         feats_2: List[NodeFeatures],
                         cost_matrix: np.ndarray,
                         same_dist_max: float = 5.0,
                         unmatched_cost: float = 0.38,
                         max_rounds: int = 5) -> list:
    """Post-Hungarian spatial reassignment to fix wrong-instance matches.

    Iteratively, for each matched pair where spatial distance > same_dist_max:
      1. Find all SG1 nodes of the same category that are closer to the SG2 node
      2. If a closer SG1 node is unmatched, steal it
      3. If a closer SG1 node is matched to a distant SG2 node, try pairwise swap

    Repeats for max_rounds or until no more improvements.
    """
    n1, n2 = len(feats_1), len(feats_2)

    # Build current assignment maps (feature-list position based)
    match_fwd = {}  # i → j
    match_rev = {}  # j → i
    pair_costs = {}
    for i, j, c in pairs:
        if c < unmatched_cost:
            match_fwd[i] = j
            match_rev[j] = i
            pair_costs[(i, j)] = c

    # Build category → feature-list-position index
    cat_to_sg1 = defaultdict(list)
    for i, f in enumerate(feats_1):
        cat_to_sg1[f.caption.lower().strip().replace('_', ' ')].append(i)

    total_steals = 0
    total_swaps = 0

    for round_num in range(max_rounds):
        n_steals = 0
        n_swaps = 0

        # Process each matched SG2 node
        for j in list(range(n2)):
            if j not in match_rev:
                continue
            i = match_rev[j]
            dist_ij = float(np.linalg.norm(feats_1[i].centroid_3d - feats_2[j].centroid_3d))
            if dist_ij <= same_dist_max:
                continue  # already close enough

            cat_j = feats_2[j].caption.lower().strip().replace('_', ' ')
            candidates = cat_to_sg1.get(cat_j, [])

            best_i2 = None
            best_dist = dist_ij
            for i2 in candidates:
                if i2 == i:
                    continue
                d = float(np.linalg.norm(feats_1[i2].centroid_3d - feats_2[j].centroid_3d))
                if d < best_dist:
                    best_dist = d
                    best_i2 = i2

            if best_i2 is None:
                continue

            if best_i2 not in match_fwd:
                # Steal: best_i2 is unmatched
                del match_fwd[i]
                match_fwd[best_i2] = j
                match_rev[j] = best_i2
                pair_costs[(best_i2, j)] = cost_matrix[best_i2, j]
                if (i, j) in pair_costs:
                    del pair_costs[(i, j)]
                n_steals += 1
            else:
                # Swap: best_i2 is matched to j2
                j2 = match_fwd[best_i2]
                old_dist = (dist_ij +
                            float(np.linalg.norm(feats_1[best_i2].centroid_3d - feats_2[j2].centroid_3d)))
                new_dist = (best_dist +
                            float(np.linalg.norm(feats_1[i].centroid_3d - feats_2[j2].centroid_3d)))
                old_cost = cost_matrix[i, j] + cost_matrix[best_i2, j2]
                new_cost = cost_matrix[best_i2, j] + cost_matrix[i, j2]

                if new_dist < old_dist * 0.85 and new_cost < old_cost * 1.2:
                    del match_fwd[i]
                    del match_fwd[best_i2]
                    match_fwd[best_i2] = j
                    match_fwd[i] = j2
                    match_rev[j] = best_i2
                    match_rev[j2] = i
                    pair_costs[(best_i2, j)] = cost_matrix[best_i2, j]
                    pair_costs[(i, j2)] = cost_matrix[i, j2]
                    if (i, j) in pair_costs:
                        del pair_costs[(i, j)]
                    if (best_i2, j2) in pair_costs:
                        del pair_costs[(best_i2, j2)]
                    n_swaps += 1

        total_steals += n_steals
        total_swaps += n_swaps
        if n_steals == 0 and n_swaps == 0:
            break

    print(f"[{_ts()}]   Spatial reassignment ({round_num+1} rounds): "
          f"{total_steals} steals, {total_swaps} swaps")

    # Rebuild pairs list
    new_pairs = []
    for (i, j), c in pair_costs.items():
        new_pairs.append((i, j, c))
    return new_pairs

@dataclass
class EdgeConsistencyReport:
    """Detailed edge consistency metrics."""
    hard: float              # strict: edge must exist in target graph
    soft: float              # proximity-based: endpoint distance < threshold
    total_checked: int       # SG1 edges with both endpoints matched
    exact_match: int         # edges found in SG2
    close_no_edge: int       # endpoints <proximity_m apart but no SG2 edge
    far_no_edge: int         # endpoints ≥proximity_m apart and no SG2 edge
    far_due_to_long_edge: int  # far-no-edge where SG1 edge itself spans ≥proximity_m
    dist_preserved: float    # fraction with |d_sg1 - d_sg2| < 1.5m
    proximity_m: float       # threshold used for soft matching
    theoretical_ceiling: float  # max soft achievable (fraction of checked edges with SG1 dist < proximity_m)


def compute_edge_consistency(match_map: dict, edges_1: list,
                             edges_2: list,
                             feats_1: List[NodeFeatures] = None,
                             feats_2: List[NodeFeatures] = None,
                             proximity_m: float = 3.0) -> EdgeConsistencyReport:
    """Compute hard and soft edge consistency with distance preservation.

    Hard: fraction of SG1 edges whose mapped endpoints have an edge in SG2.
    Soft: fraction where endpoints either have an edge OR are within proximity_m.
    Distance preservation: fraction where |d_sg1 - d_sg2| < 1.5m.
    Theoretical ceiling: fraction of checked SG1 edges spanning < proximity_m.
    """
    if not edges_1:
        return EdgeConsistencyReport(
            hard=1.0, soft=1.0, total_checked=0, exact_match=0,
            close_no_edge=0, far_no_edge=0, far_due_to_long_edge=0,
            dist_preserved=1.0, proximity_m=proximity_m,
            theoretical_ceiling=1.0)

    edge_set_2 = set()
    for e in edges_2:
        edge_set_2.add(frozenset([e['node1_idx'], e['node2_idx']]))

    # Build position lookups
    pos_1 = {}
    if feats_1:
        for f in feats_1:
            pos_1[f.idx] = f.centroid_3d
    pos_2 = {}
    if feats_2:
        for f in feats_2:
            pos_2[f.idx] = f.centroid_3d

    exact = 0
    close = 0
    far = 0
    far_long = 0        # far-no-edge where SG1 edge ≥ proximity_m
    n_preserved = 0      # distance well-preserved
    n_with_dist = 0      # edges where we could compute both distances
    n_short_sg1 = 0      # SG1 edges < proximity_m (for theoretical ceiling)
    total = 0

    for e in edges_1:
        n1, n2 = e['node1_idx'], e['node2_idx']
        mapped_n1 = match_map.get(n1)
        mapped_n2 = match_map.get(n2)
        if mapped_n1 is None or mapped_n2 is None:
            continue
        total += 1

        # SG1 edge distance
        d_sg1 = None
        if n1 in pos_1 and n2 in pos_1:
            d_sg1 = float(np.linalg.norm(pos_1[n1] - pos_1[n2]))
            if d_sg1 < proximity_m:
                n_short_sg1 += 1

        # SG2 mapped endpoint distance
        d_sg2 = None
        if mapped_n1 in pos_2 and mapped_n2 in pos_2:
            d_sg2 = float(np.linalg.norm(pos_2[mapped_n1] - pos_2[mapped_n2]))

        # Distance preservation
        if d_sg1 is not None and d_sg2 is not None:
            n_with_dist += 1
            if abs(d_sg1 - d_sg2) < 1.5:
                n_preserved += 1

        # Edge consistency classification
        if frozenset([mapped_n1, mapped_n2]) in edge_set_2:
            exact += 1
        elif d_sg2 is not None and d_sg2 < proximity_m:
            close += 1
        else:
            far += 1
            if d_sg1 is not None and d_sg1 >= proximity_m:
                far_long += 1

    hard = exact / max(total, 1)
    soft = (exact + close) / max(total, 1)
    dist_pres = n_preserved / max(n_with_dist, 1)
    ceiling = n_short_sg1 / max(total, 1)

    return EdgeConsistencyReport(
        hard=hard, soft=soft, total_checked=total, exact_match=exact,
        close_no_edge=close, far_no_edge=far, far_due_to_long_edge=far_long,
        dist_preserved=dist_pres, proximity_m=proximity_m,
        theoretical_ceiling=ceiling)


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

    snaps_1 = build_rich_snapshots(sg1_data, label="SG1")
    snaps_2 = build_rich_snapshots(sg2_data, label="SG2")

    # Build adjacency, edge relations, and compute neighborhood features
    adj_1 = build_adjacency(sg1_data)
    adj_2 = build_adjacency(sg2_data)
    edge_rels_1 = build_edge_relations(sg1_data)
    edge_rels_2 = build_edge_relations(sg2_data)
    all_cats = sorted(set(
        f.caption.lower().strip().replace('_', ' ')
        for f in feats_1 + feats_2))
    cat_to_bin = {c: i for i, c in enumerate(all_cats)}
    compute_neighborhood_features(feats_1, adj_1, edge_rels_1, cat_to_bin, label="SG1")
    compute_neighborhood_features(feats_2, adj_2, edge_rels_2, cat_to_bin, label="SG2")

    # Build soft room similarity table
    room_sim = build_room_sim_table()
    rooms_1 = set(f.room_caption for f in feats_1 if f.room_caption)
    rooms_2 = set(f.room_caption for f in feats_2 if f.room_caption)
    print(f"[{_ts()}] Room hierarchy: SG1 uses {len(rooms_1)} rooms "
          f"{sorted(rooms_1)}, SG2 uses {len(rooms_2)} rooms {sorted(rooms_2)}")

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
                centroid_3d=f.centroid_3d.tolist(), room_idx=f.room_idx,
                room_caption=f.room_caption)))
        for f in feats_2:
            unmatched.append(asdict(UnmatchedNode(
                sg_source="sg2", node_idx=f.idx, caption=f.caption,
                change_type=ChangeType.ADDED.value,
                centroid_3d=f.centroid_3d.tolist(), room_idx=f.room_idx,
                room_caption=f.room_caption)))
        return MatchReport(
            sg1_path=sg1_path, sg2_path=sg2_path,
            sg1_num_nodes=n1, sg2_num_nodes=n2,
            sg1_num_edges=len(sg1_data['edges']),
            sg2_num_edges=len(sg2_data['edges']),
            matches=[], unmatched=unmatched,
            edge_consistency_global=0.0,
            edge_consistency_soft=0.0,
            edge_consistency_detail={},
            summary={t.value: 0 for t in ChangeType})

    # Auto-compute max_match_dist if not set
    if weights.max_match_dist < 0:
        auto_max_dist = compute_auto_max_match_dist(feats_1, feats_2)
        weights.max_match_dist = auto_max_dist
        print(f"[{_ts()}] Auto max_match_dist: {auto_max_dist:.1f}m")
    else:
        print(f"[{_ts()}] Max match dist: {weights.max_match_dist:.1f}m (preset)")

    # Build and solve cost matrix
    cost_matrix = build_cost_matrix(feats_1, feats_2, weights, snaps_1, snaps_2,
                                    room_sim=room_sim)

    if refine_iters > 0:
        print(f"[{_ts()}] Running iterative neighborhood consensus "
              f"({refine_iters} rounds) ...")
        pairs = iterative_neighborhood_consensus(
            cost_matrix, feats_1, feats_2, adj_1, adj_2,
            n_iters=refine_iters, unmatched_cost=weights.unmatched_cost)
    else:
        pairs = solve_assignment(cost_matrix, n1, n2, weights.unmatched_cost)

    # Post-match spatial reassignment: fix wrong-instance matches
    print(f"[{_ts()}] Running spatial reassignment ...")
    pairs = spatial_reassignment(
        pairs, feats_1, feats_2, cost_matrix,
        same_dist_max=thresholds.same_dist_max,
        unmatched_cost=weights.unmatched_cost)

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
        # Skip dummy pairs from spatial_reassignment
        if j >= n2 or i >= n1:
            continue
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
            same_room=(fa.room_caption == fb.room_caption
                       if fa.room_caption and fb.room_caption else False),
            sg1_room=fa.room_caption,
            sg2_room=fb.room_caption,
            edge_consistency=0.0,
            confidence=confidence,
        )))

        if verbose:
            print(f"  {fa.caption:23s} {fb.caption:23s} {c:6.3f} "
                  f"{clip_sim:6.3f} {chist_sim:5.3f} {dist_3d:6.2f}m "
                  f"{change_type.value:>10s} {confidence:5.3f}")

    # Post-classification: downgrade ambiguous MOVED to UNCERTAIN
    # If a "MOVED" object has many same-category alternatives closer to the
    # SG2 node, the match is likely a wrong-instance Hungarian artifact
    cat_to_sg1_feats = defaultdict(list)
    for f in feats_1:
        cat_to_sg1_feats[f.caption.lower().strip().replace('_', ' ')].append(f)

    n_downgraded = 0
    for m in matches:
        if m['change_type'] != 'MOVED':
            continue
        cat = m['sg2_caption'].lower().strip().replace('_', ' ')
        sg2_c = None
        for f in feats_2:
            if f.idx == m['sg2_idx']:
                sg2_c = f.centroid_3d
                break
        if sg2_c is None:
            continue
        # Count same-category SG1 nodes closer than this match
        same_cat_sg1 = cat_to_sg1_feats.get(cat, [])
        n_closer = sum(1 for f1 in same_cat_sg1
                       if float(np.linalg.norm(f1.centroid_3d - sg2_c))
                       < m['spatial_dist_3d'])
        # If there are ≥3 closer same-category alternatives, the match
        # is ambiguous — the "movement" is likely a wrong-instance artifact
        if n_closer >= 3:
            m['change_type'] = 'UNCERTAIN'
            m['confidence'] = round(m['confidence'] * 0.5, 3)
            n_downgraded += 1

    if n_downgraded > 0:
        print(f"[{_ts()}]   Downgraded {n_downgraded} ambiguous MOVED → UNCERTAIN")

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
                centroid_3d=f.centroid_3d.tolist(), room_idx=f.room_idx,
                room_caption=f.room_caption)))

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
                centroid_3d=f.centroid_3d.tolist(), room_idx=f.room_idx,
                room_caption=f.room_caption)))

    # Edge consistency — computed on accepted matches only (post-classification)
    print(f"[{_ts()}] Computing edge consistency ({len(matches)} matches, "
          f"{len(sg1_data['edges'])}+{len(sg2_data['edges'])} edges) ...")
    match_map_node = {}
    for m in matches:
        match_map_node[m['sg1_idx']] = m['sg2_idx']
    ec_report = compute_edge_consistency(
        match_map_node, sg1_data['edges'], sg2_data['edges'],
        feats_1=feats_1, feats_2=feats_2, proximity_m=3.0)
    edge_relation_acc = compute_edge_relation_accuracy(
        match_map_node, sg1_data['edges'], sg2_data['edges'])

    # Backfill edge_consistency into match results
    for m in matches:
        m['edge_consistency'] = round(ec_report.hard, 4)

    if verbose:
        print(f"\nEdge consistency (hard):  {ec_report.hard:.2%} "
              f"({ec_report.exact_match}/{ec_report.total_checked} edges)")
        print(f"Edge consistency (soft):  {ec_report.soft:.2%} "
              f"({ec_report.exact_match + ec_report.close_no_edge}/"
              f"{ec_report.total_checked}, "
              f"proximity<{ec_report.proximity_m}m)")
        print(f"  Breakdown: {ec_report.exact_match} exact, "
              f"{ec_report.close_no_edge} close-no-edge, "
              f"{ec_report.far_no_edge} far-no-edge "
              f"({ec_report.far_due_to_long_edge} due to long SG1 edges)")
        print(f"Distance preserved:      {ec_report.dist_preserved:.2%}")
        print(f"Theoretical ceiling:     {ec_report.theoretical_ceiling:.2%} "
              f"(fraction of checked SG1 edges < {ec_report.proximity_m}m)")
        print(f"Edge relation accuracy:  {edge_relation_acc:.2%}")

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
        edge_consistency_global=round(ec_report.hard, 4),
        edge_consistency_soft=round(ec_report.soft, 4),
        edge_consistency_detail=asdict(ec_report),
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
    parser.add_argument('--w-rel-hist', type=float, default=None)
    parser.add_argument('--unmatched-cost', type=float, default=None)
    parser.add_argument('--scene-scale', type=float, default=-1.0,
                        help='Scene scale in meters (-1 = auto from data)')
    parser.add_argument('--max-match-dist', type=float, default=None,
                        help='Hard spatial gate: reject matches beyond this distance (m). '
                             'Default: auto from data or mode preset.')

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
                       ('w_rel_hist', 'w_rel_hist'),
                       ('unmatched_cost', 'unmatched_cost')]:
        val = getattr(args, flag)
        if val is not None:
            setattr(weights, attr, val)
    weights.scene_scale = args.scene_scale
    if args.max_match_dist is not None:
        weights.max_match_dist = args.max_match_dist
    rel_pct = weights.w_rel_hist
    neigh_pct = weights.w_nch + weights.w_neighbor_clip + rel_pct
    pos_pct = weights.w_spatial + weights.w_room
    id_pct = 1.0 - pos_pct - neigh_pct
    print(f"[{_ts()}] Mode: {args.mode} | Position: {pos_pct:.0%} | "
          f"Neighborhood: {neigh_pct:.0%} (rel_hist: {rel_pct:.0%}) | "
          f"Identity: {id_pct:.0%} | "
          f"Refine iters: {args.refine_iters}")

    # Mode-aware threshold defaults
    if args.mode == 'dynamic':
        default_same_dist = 3.0     # tighter than before (spatial gating handles outliers)
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
    ec_detail = report.edge_consistency_detail
    print(f"  {'Edge consistency (hard)':<36s} {report.edge_consistency_global:>25.1%}")
    print(f"  {'Edge consistency (soft <3m)':<36s} {report.edge_consistency_soft:>25.1%}")
    if ec_detail:
        t = ec_detail.get('total_checked', 0)
        ex = ec_detail.get('exact_match', 0)
        cl = ec_detail.get('close_no_edge', 0)
        fa = ec_detail.get('far_no_edge', 0)
        fl = ec_detail.get('far_due_to_long_edge', 0)
        dp = ec_detail.get('dist_preserved', 0)
        ceil = ec_detail.get('theoretical_ceiling', 0)
        print(f"    {'└ exact / close / far':<34s} {ex:>6d} / {cl:>5d} / {fa:>5d}  (of {t})")
        if fa > 0:
            print(f"    {'└ far: long SG1 edges (≥3m)':<34s} {fl:>6d} / {fa}")
        print(f"  {'Distance preservation (<1.5m)':<36s} {dp:>25.1%}")
        print(f"  {'Theoretical soft ceiling':<36s} {ceil:>25.1%}")
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

    # ---- Room-level matching breakdown ----
    from collections import Counter as _Counter
    same_room_count = sum(1 for m in report.matches
                         if m['change_type'] in ('SAME', 'MOVED')
                         and m.get('same_room', False))
    total_accepted = sum(1 for m in report.matches
                         if m['change_type'] in ('SAME', 'MOVED'))
    room_pct = 100 * same_room_count / total_accepted if total_accepted else 0
    print(f"\n  Room-Level Analysis:")
    print(f"  {thin}")
    print(f"  {'Same-room matches':<36s} {same_room_count:>5d} / {total_accepted} ({room_pct:.1f}%)")
    # Per-room distribution of matched nodes
    room_same = _Counter()
    room_moved = _Counter()
    for m in report.matches:
        r_label = m.get('sg1_room') or '(unknown)'
        if m['change_type'] == 'SAME':
            room_same[r_label] += 1
        elif m['change_type'] == 'MOVED':
            room_moved[r_label] += 1
    all_room_labels = sorted(set(list(room_same.keys()) + list(room_moved.keys())))
    if all_room_labels:
        print(f"  {'Room':<20s} {'SAME':>8s} {'MOVED':>8s} {'Total':>8s}")
        print(f"  {thin}")
        for rl in all_room_labels:
            s, mv = room_same.get(rl, 0), room_moved.get(rl, 0)
            print(f"  {rl:<20s} {s:>8d} {mv:>8d} {s+mv:>8d}")

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
    rel_pct = weights.w_rel_hist
    neigh_pct = weights.w_nch + weights.w_neighbor_clip + rel_pct
    pos_pct = weights.w_spatial + weights.w_room
    id_pct = 1.0 - pos_pct - neigh_pct
    print(f"\n  Config:")
    print(f"  {'Weight split':<28s}  Position {pos_pct:.0%} | "
          f"Neighborhood {neigh_pct:.0%} (rel_hist: {rel_pct:.0%}) | "
          f"Snapshot {weights.w_snapshot:.0%} | Identity {id_pct:.0%}")
    print(f"  {'Thresholds':<28s}  same_dist≤{thresholds.same_dist_max}m  "
          f"same_clip≥{thresholds.same_clip_min}  "
          f"moved_clip≥{thresholds.moved_clip_min}")
    print(f"  {'Spatial gate':<28s}  max_match_dist={weights.max_match_dist:.1f}m")
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
