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

    Uses 1 - exp(-d²/2σ²) with σ = scene_scale/2.5 so that:
      1m → ~0.05,  2m → ~0.18,  3m → ~0.36,  5m → ~0.71,  8m → ~0.96
    """
    dist = float(np.linalg.norm(a.centroid_3d - b.centroid_3d))
    sigma = scene_scale / 2.5
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


# ---------------------------------------------------------------------------
# Cost matrix & Hungarian assignment
# ---------------------------------------------------------------------------

@dataclass
class MatchWeights:
    """Weights for combining cost components.

    Rebalanced for instance-level discrimination.  Color histogram is
    the strongest same-category discriminator (gap=0.229 vs CLIP's 0.066).
    """
    w_spatial: float = 0.30       # primary geometric signal
    w_color: float = 0.20         # HSV histogram — strongest instance signal
    w_clip_visual: float = 0.14   # useful cross-category, weak intra-category
    w_snapshot: float = 0.10      # edge context
    w_label: float = 0.08         # coarse category filter
    w_shape: float = 0.08         # PCA geometry
    w_clip_text: float = 0.05     # semantic backup
    w_room: float = 0.05          # room hierarchy
    unmatched_cost: float = 0.42
    scene_scale: float = 5.0


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
        if i % report_interval == 0 and i > 0:
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

            cost[i, j] = (weights.w_clip_visual * c_vis +
                          weights.w_spatial * c_spa +
                          weights.w_label * c_lbl +
                          weights.w_clip_text * c_txt +
                          weights.w_room * c_rm +
                          weights.w_snapshot * c_snap +
                          weights.w_color * c_color +
                          weights.w_shape * c_shape)

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

def load_scene_graph(filepath: str) -> dict:
    """Load a scene graph from a .pkl file."""
    print(f"[{_ts()}] Loading {filepath.split('/')[-1]} ...")
    with open(filepath, 'rb') as f:
        save_data = pickle.load(f)
    sg = save_data['scenegraph']
    print(f"[{_ts()}]   -> {len(sg['nodes'])} nodes, {len(sg['edges'])} edges")
    return sg


def match_scene_graphs(sg1_path: str, sg2_path: str,
                       weights: MatchWeights = None,
                       thresholds: ChangeThresholds = None,
                       verbose: bool = False) -> MatchReport:
    """Full matching pipeline between two saved scene graphs."""
    if weights is None:
        weights = MatchWeights()
    if thresholds is None:
        thresholds = ChangeThresholds()

    sg1_data = load_scene_graph(sg1_path)
    sg2_data = load_scene_graph(sg2_path)

    feats_1 = extract_node_features(sg1_data, label="SG1")
    feats_2 = extract_node_features(sg2_data, label="SG2")

    snaps_1 = build_node_edge_snapshots(sg1_data, label="SG1")
    snaps_2 = build_node_edge_snapshots(sg2_data, label="SG2")

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

    # Unmatched nodes
    unmatched = []
    for i, f in enumerate(feats_1):
        if i not in matched_sg1:
            unmatched.append(asdict(UnmatchedNode(
                sg_source="sg1", node_idx=f.idx, caption=f.caption,
                change_type=ChangeType.REMOVED.value,
                centroid_3d=f.centroid_3d.tolist(), room_idx=f.room_idx)))

    for j, f in enumerate(feats_2):
        if j not in matched_sg2:
            unmatched.append(asdict(UnmatchedNode(
                sg_source="sg2", node_idx=f.idx, caption=f.caption,
                change_type=ChangeType.ADDED.value,
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
    summary[ChangeType.ADDED.value] = sum(
        1 for u in unmatched if u['change_type'] == ChangeType.ADDED.value)
    summary[ChangeType.REMOVED.value] = sum(
        1 for u in unmatched if u['change_type'] == ChangeType.REMOVED.value)

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

    # Weight overrides
    parser.add_argument('--w-spatial', type=float, default=0.30)
    parser.add_argument('--w-color', type=float, default=0.20)
    parser.add_argument('--w-clip-visual', type=float, default=0.14)
    parser.add_argument('--w-snapshot', type=float, default=0.10)
    parser.add_argument('--w-label', type=float, default=0.08)
    parser.add_argument('--w-shape', type=float, default=0.08)
    parser.add_argument('--w-clip-text', type=float, default=0.05)
    parser.add_argument('--w-room', type=float, default=0.05)
    parser.add_argument('--unmatched-cost', type=float, default=0.42)
    parser.add_argument('--scene-scale', type=float, default=5.0)

    # Change detection thresholds
    parser.add_argument('--same-clip-min', type=float, default=0.75)
    parser.add_argument('--same-dist-max', type=float, default=2.0)
    parser.add_argument('--moved-clip-min', type=float, default=0.70)

    args = parser.parse_args()

    weights = MatchWeights(
        w_spatial=args.w_spatial,
        w_color=args.w_color,
        w_clip_visual=args.w_clip_visual,
        w_snapshot=args.w_snapshot,
        w_label=args.w_label,
        w_shape=args.w_shape,
        w_clip_text=args.w_clip_text,
        w_room=args.w_room,
        unmatched_cost=args.unmatched_cost,
        scene_scale=args.scene_scale,
    )
    thresholds = ChangeThresholds(
        same_clip_min=args.same_clip_min,
        same_dist_max=args.same_dist_max,
        moved_clip_min=args.moved_clip_min,
        cost_reject=args.unmatched_cost,
    )

    report = match_scene_graphs(
        args.sg1, args.sg2, weights, thresholds, verbose=args.verbose)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  MATCHING SUMMARY")
    print(f"{'='*60}")
    print(f"  SG1: {report.sg1_num_nodes} nodes, {report.sg1_num_edges} edges")
    print(f"  SG2: {report.sg2_num_nodes} nodes, {report.sg2_num_edges} edges")
    print(f"  Edge consistency: {report.edge_consistency_global:.1%}")
    print(f"")
    for change_type, count in report.summary.items():
        print(f"    {change_type:12s}: {count}")

    # Per-category breakdown for SAME matches
    same_matches = [m for m in report.matches if m['change_type'] == 'SAME']
    if same_matches:
        from collections import Counter
        cat_counts = Counter(m['sg1_caption'] for m in same_matches)
        same_dists = [m['spatial_dist_3d'] for m in same_matches]
        same_clips = [m['clip_visual_sim'] for m in same_matches]
        same_colors = [m['color_hist_sim'] for m in same_matches]
        same_confs = [m['confidence'] for m in same_matches]
        print(f"\n  SAME quality (n={len(same_matches)}):")
        print(f"    Avg CLIP sim:   {sum(same_clips)/len(same_clips):.3f}")
        print(f"    Avg color sim:  {sum(same_colors)/len(same_colors):.3f}")
        print(f"    Avg distance:   {sum(same_dists)/len(same_dists):.2f}m")
        print(f"    Avg confidence: {sum(same_confs)/len(same_confs):.3f}")
        print(f"    Top categories: {cat_counts.most_common(8)}")

    # MOVED breakdown
    moved_matches = [m for m in report.matches if m['change_type'] == 'MOVED']
    if moved_matches:
        moved_dists = [m['spatial_dist_3d'] for m in moved_matches]
        print(f"\n  MOVED details (n={len(moved_matches)}):")
        print(f"    Avg distance: {sum(moved_dists)/len(moved_dists):.2f}m")
        print(f"    Range: {min(moved_dists):.2f}m - {max(moved_dists):.2f}m")

    print(f"{'='*60}")

    if args.output:
        result = asdict(report)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nFull report saved to {args.output}")


if __name__ == '__main__':
    main()
