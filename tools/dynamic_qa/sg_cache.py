from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SerializedSceneGraph = Dict[str, Any]


@dataclass
class GlobalSGSave:
    """Loaded output of SG_Nav_Agent.save_global_scenegraph()."""

    path: Path
    save_data: Dict[str, Any]

    @property
    def scenegraph(self) -> SerializedSceneGraph:
        return self.save_data["scenegraph"]

    @property
    def scene_id(self) -> Optional[str]:
        return self.save_data.get("scene_id")

    @property
    def episode_id(self) -> Optional[str]:
        return self.save_data.get("episode_id")


def load_global_sg_pkl(path: str | Path) -> GlobalSGSave:
    path = Path(path)
    with path.open("rb") as f:
        save_data = pickle.load(f)
    return GlobalSGSave(path=path, save_data=save_data)


def iter_nodes(sg: SerializedSceneGraph) -> Iterable[Dict[str, Any]]:
    return sg.get("nodes", [])


def iter_edges(sg: SerializedSceneGraph) -> Iterable[Dict[str, Any]]:
    return sg.get("edges", [])


def node_caption(node: Dict[str, Any]) -> str:
    return str(node.get("caption", ""))


def node_room_caption(sg: SerializedSceneGraph, node: Dict[str, Any]) -> Optional[str]:
    ridx = node.get("room_idx")
    rooms = sg.get("room_nodes") or []
    if ridx is None or ridx < 0 or ridx >= len(rooms):
        return None
    return rooms[ridx].get("caption")


def _as_np_xyz(points: Any) -> Optional[np.ndarray]:
    if points is None:
        return None
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None

    # Heuristic fix: some saved SG pkls store points as (x, z, y) not (x, y, z).
    # Detect by checking magnitudes: height (Y) is usually small (~[-1,3]), while Z spans many meters.
    try:
        p95_1 = float(np.nanpercentile(np.abs(arr[:, 1]), 95))
        p95_2 = float(np.nanpercentile(np.abs(arr[:, 2]), 95))
        if p95_1 > 5.0 and p95_2 < 5.0:
            # SG-Nav stores points as (X_horiz, Y_horiz, Z_vert) but Habitat
            # uses (X, Y_vert, Z_horiz). Swap col 1↔2 to get Y-up convention.
            arr = arr[:, [0, 2, 1]]
    except Exception:
        pass

    return arr


def node_pcd_points(node: Dict[str, Any]) -> Optional[np.ndarray]:
    obj = node.get("object") or {}
    return _as_np_xyz(obj.get("pcd_points"))


def node_bbox_points(node: Dict[str, Any]) -> Optional[np.ndarray]:
    obj = node.get("object") or {}
    return _as_np_xyz(obj.get("bbox_points"))


def node_centroid_world(node: Dict[str, Any]) -> Optional[np.ndarray]:
    pts = node_pcd_points(node)
    if pts is None or len(pts) == 0:
        pts = node_bbox_points(node)
    if pts is None or len(pts) == 0:
        return None
    return pts.mean(axis=0)


def node_aabb_world(node: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    pts = node_pcd_points(node)
    if pts is None or len(pts) == 0:
        pts = node_bbox_points(node)
    if pts is None or len(pts) == 0:
        return None
    return pts.min(axis=0), pts.max(axis=0)


def node_top_surface_z(node: Dict[str, Any]) -> Optional[float]:
    aabb = node_aabb_world(node)
    if aabb is None:
        return None
    _mn, mx = aabb
    # After _as_np_xyz() swap, data is [X, Y_up, Z] — index 1 = height
    return float(mx[1])


def candidate_support_nodes(
    sg: SerializedSceneGraph,
    include_captions: Optional[Sequence[str]] = None,
    exclude_captions: Optional[Sequence[str]] = None,
    require_pcd: bool = True,
) -> List[Dict[str, Any]]:
    """Heuristic filter for nodes that can act as placement supports."""

    include = set(c.lower() for c in include_captions) if include_captions else None
    exclude = set(c.lower() for c in exclude_captions) if exclude_captions else set()

    supports: List[Dict[str, Any]] = []
    for n in iter_nodes(sg):
        cap = node_caption(n).lower()
        if include is not None and cap not in include:
            continue
        if cap in exclude:
            continue
        if require_pcd:
            pts = node_pcd_points(n)
            if pts is None or len(pts) == 0:
                continue
        supports.append(n)

    # Prefer larger supports by AABB footprint
    def score(n: Dict[str, Any]) -> float:
        aabb = node_aabb_world(n)
        if aabb is None:
            return 0.0
        mn, mx = aabb
        dx = float(mx[0] - mn[0])
        dz = float(mx[2] - mn[2])
        return dx * dz

    supports.sort(key=score, reverse=True)
    return supports


def build_support_graph_text(
    sg: SerializedSceneGraph,
    node_idxs: Sequence[int],
    max_edges: int = 200,
    include_snapshot_meta: bool = True,
) -> str:
    """Compact text view for prompting a selector / QA generator."""

    nodes = sg.get("nodes") or []
    idx_set = set(node_idxs)

    lines: List[str] = []
    lines.append("NODES:")
    for idx in node_idxs:
        if idx < 0 or idx >= len(nodes):
            continue
        n = nodes[idx]
        room = node_room_caption(sg, n)
        c = node_centroid_world(n)
        c_str = "?" if c is None else f"[{c[0]:.2f},{c[1]:.2f},{c[2]:.2f}]"
        if room:
            lines.append(f"- ({idx}) {n.get('caption')} | room={room} | centroid={c_str}")
        else:
            lines.append(f"- ({idx}) {n.get('caption')} | centroid={c_str}")

    edges = sg.get("edges") or []
    lines.append("EDGES:")
    used = 0
    for e in edges:
        if used >= max_edges:
            break
        i = e.get("node1_idx")
        j = e.get("node2_idx")
        if i not in idx_set and j not in idx_set:
            continue
        rel = e.get("relation")
        snap = ""
        if include_snapshot_meta and ("snapshot_step" in e or "snapshot_b64" in e):
            meta = []
            if e.get("snapshot_step") is not None:
                meta.append(f"step={e.get('snapshot_step')}")
            if e.get("snapshot_frame_idx") is not None:
                meta.append(f"frame={e.get('snapshot_frame_idx')}")
            if meta:
                snap = " [snapshot: " + ", ".join(meta) + "]"
            else:
                snap = " [snapshot]"
        lines.append(f"- ({i}) --{rel}--> ({j}){snap}")
        used += 1

    return "\n".join(lines)
