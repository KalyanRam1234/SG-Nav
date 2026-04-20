from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .sg_cache import SerializedSceneGraph, candidate_support_nodes, iter_nodes, node_caption


@dataclass
class SupportChoice:
    node_idx: int
    caption: str
    room_idx: Optional[int] = None


DEFAULT_SUPPORT_TYPES = [
    "table",
    "counter",
    "desk",
    "sofa",
    "bed",
    "sink",
    "cabinet",
    "drawers",
    "shelf",
    "nightstand",
    "chair",
]

# Objects that should NEVER be used as placement surfaces
NON_SURFACE_OBJECTS = {
    "rug", "door", "window", "picture", "mirror", "plant", "lamp",
    "blanket", "curtain", "pillow", "cushion", "towel", "toilet",
    "shower", "bathtub", "ceiling fan", "wall",
}


OBJECT_TO_SUPPORT_TYPES = {
    "phone": ["counter", "table", "desk", "nightstand", "drawers", "sofa", "bed", "cabinet", "shelf"],
    "cell phone": ["counter", "table", "desk", "nightstand", "drawers", "sofa", "bed", "cabinet", "shelf"],
    "remote": ["sofa", "table", "counter", "bed", "nightstand", "drawers", "cabinet"],
    "tv remote": ["sofa", "table", "counter", "bed", "nightstand", "drawers", "cabinet"],
    "laptop": ["desk", "table", "bed", "counter", "drawers"],
    "mug": ["table", "counter", "sink", "desk", "drawers", "cabinet", "nightstand", "shelf"],
    "cup": ["table", "counter", "sink", "desk", "drawers", "cabinet", "nightstand", "shelf"],
    "bottle": ["table", "counter", "desk", "drawers", "cabinet", "shelf"],
    "book": ["table", "desk", "bed", "shelf", "nightstand", "drawers", "cabinet"],
    "keys": ["table", "counter", "desk", "nightstand", "drawers", "cabinet", "shelf"],
}


def _room_idx(node: Dict[str, Any]) -> Optional[int]:
    ridx = node.get("room_idx")
    return int(ridx) if ridx is not None else None


def get_all_support_candidates(
    sg: SerializedSceneGraph,
    inserted_object_category: str,
    support_types_override: Optional[Sequence[str]] = None,
) -> List[SupportChoice]:
    """Return all support candidates ranked by priority (no pairing logic)."""
    obj = inserted_object_category.lower().strip()
    support_types = (
        [s.lower() for s in support_types_override]
        if support_types_override
        else OBJECT_TO_SUPPORT_TYPES.get(obj, DEFAULT_SUPPORT_TYPES)
    )

    nodes = list(iter_nodes(sg))
    idx_by_caption: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for i, n in enumerate(nodes):
        cap = node_caption(n).lower()
        idx_by_caption.setdefault(cap, []).append((i, n))

    pools: List[List[Tuple[int, Dict[str, Any]]]] = []
    for cap in support_types:
        if cap in idx_by_caption:
            pools.append(idx_by_caption[cap])

    if not pools:
        supports = candidate_support_nodes(sg, require_pcd=True)
        surface_supports = [
            n for n in supports if node_caption(n).lower() not in NON_SURFACE_OBJECTS
        ]
        if not surface_supports:
            return []
        pools = [[(nodes.index(n), n) for n in surface_supports[:50] if n in nodes]]

    flat: List[Tuple[int, Dict[str, Any]]] = []
    for p in pools:
        flat.extend(p)

    seen: set = set()
    result: List[SupportChoice] = []
    for idx, n in flat:
        if idx in seen:
            continue
        seen.add(idx)
        result.append(SupportChoice(node_idx=idx, caption=node_caption(n), room_idx=_room_idx(n)))
    return result


def select_support_pair(
    sg: SerializedSceneGraph,
    inserted_object_category: str,
    prefer_distinct_rooms: bool = True,
    support_types_override: Optional[Sequence[str]] = None,
) -> Tuple[SupportChoice, SupportChoice]:
    """Pick two plausible supports (A,B) from a saved scene graph.

    This is a heuristic baseline (no LLM). It prefers large supports and,
    when possible, chooses supports from different rooms.
    """

    obj = inserted_object_category.lower().strip()
    support_types = (
        [s.lower() for s in support_types_override]
        if support_types_override
        else OBJECT_TO_SUPPORT_TYPES.get(obj, DEFAULT_SUPPORT_TYPES)
    )

    # Nodes are usually single-word categories; treat as exact match first.
    nodes = list(iter_nodes(sg))
    idx_by_caption = {}
    for i, n in enumerate(nodes):
        cap = node_caption(n).lower()
        idx_by_caption.setdefault(cap, []).append((i, n))

    # Candidate pools in priority order.
    pools: List[List[Tuple[int, Dict[str, Any]]]] = []
    for cap in support_types:
        if cap in idx_by_caption:
            pools.append(idx_by_caption[cap])

    if not pools:
        # Fallback: take the largest supports, but exclude non-surface objects
        print(f"[DynamicQA] WARNING: No preferred support types ({support_types}) found "
              f"for '{inserted_object_category}'. Falling back to largest surface objects.")
        supports = candidate_support_nodes(sg, require_pcd=True)
        # Filter out objects that can't act as surfaces
        surface_supports = []
        for n in supports:
            cap = node_caption(n).lower()
            if cap not in NON_SURFACE_OBJECTS:
                surface_supports.append(n)
        if not surface_supports:
            raise ValueError(
                f"No suitable support surfaces found for '{inserted_object_category}'. "
                f"Scene has: {[node_caption(n) for n in list(iter_nodes(sg))[:20]]}. "
                f"None are valid placement surfaces."
            )
        pools = [[(nodes.index(n), n) for n in surface_supports[:50] if n in nodes]]

    flat: List[Tuple[int, Dict[str, Any]]] = []
    for p in pools:
        flat.extend(p)

    # Deduplicate by node index, keep order.
    seen = set()
    dedup: List[Tuple[int, Dict[str, Any]]] = []
    for idx, n in flat:
        if idx in seen:
            continue
        seen.add(idx)
        dedup.append((idx, n))

    if len(dedup) < 2:
        raise ValueError(
            f"Not enough support candidates for '{inserted_object_category}'. "
            f"Found {len(dedup)} candidates."
        )

    a_idx, a_node = dedup[0]
    a = SupportChoice(node_idx=a_idx, caption=node_caption(a_node), room_idx=_room_idx(a_node))

    # Pick B: different room if possible.
    b: Optional[SupportChoice] = None
    if prefer_distinct_rooms and a.room_idx is not None:
        for b_idx, b_node in dedup[1:]:
            ridx = _room_idx(b_node)
            if ridx is not None and ridx != a.room_idx:
                b = SupportChoice(node_idx=b_idx, caption=node_caption(b_node), room_idx=ridx)
                break

    if b is None:
        b_idx, b_node = dedup[1]
        b = SupportChoice(node_idx=b_idx, caption=node_caption(b_node), room_idx=_room_idx(b_node))

    return a, b
