from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .sg_cache import SerializedSceneGraph, iter_edges, node_room_caption


def _neighbors_for_node(sg: SerializedSceneGraph, node_idx: int) -> List[Tuple[int, str, Dict[str, Any]]]:
    out: List[Tuple[int, str, Dict[str, Any]]] = []
    for e in iter_edges(sg):
        i = e.get("node1_idx")
        j = e.get("node2_idx")
        rel = e.get("relation")
        if i == node_idx and j is not None:
            out.append((int(j), str(rel), e))
        elif j == node_idx and i is not None:
            out.append((int(i), str(rel), e))
    return out


def generate_qa_pairs(
    sg: SerializedSceneGraph,
    inserted_object: str,
    support_a_idx: int,
    support_b_idx: int,
    seed: int = 0,
    max_pairs: int = 8,
) -> List[Dict[str, Any]]:
    """Generate temporal + compositional QA leveraging advanced SG context.

    The returned QA entries are time-specific ("A" or "B").
    """

    rng = random.Random(seed)
    nodes = sg.get("nodes") or []

    def cap(idx: int) -> str:
        if idx < 0 or idx >= len(nodes):
            return f"node_{idx}"
        return str(nodes[idx].get("caption"))

    obj = inserted_object
    a_cap = cap(support_a_idx)
    b_cap = cap(support_b_idx)

    qa: List[Dict[str, Any]] = []

    # 1) Where initially / after
    qa.append(
        {
            "type": "where_initial",
            "time": "A",
            "question": f"Where is the {obj} initially placed?",
            "answer": a_cap,
            "support_node_idx": support_a_idx,
        }
    )
    qa.append(
        {
            "type": "where_after_move",
            "time": "B",
            "question": f"Where is the {obj} after it was moved?",
            "answer": b_cap,
            "support_node_idx": support_b_idx,
        }
    )

    # 2) Yes/No temporal checks (same question, different time)
    yn_q = f"Did I leave my {obj} on the {a_cap}?"
    qa.append(
        {
            "type": "did_leave_temporal",
            "time": "A",
            "question": yn_q,
            "answer": "yes",
            "support_node_idx": support_a_idx,
        }
    )
    qa.append(
        {
            "type": "did_leave_temporal",
            "time": "B",
            "question": yn_q,
            "answer": "no",
            "support_node_idx": support_a_idx,
        }
    )

    # 3) Room-conditioned question (uses room_nodes mapping)
    room_a = node_room_caption(sg, nodes[support_a_idx]) if support_a_idx < len(nodes) else None
    room_b = node_room_caption(sg, nodes[support_b_idx]) if support_b_idx < len(nodes) else None
    if room_a:
        qa.append(
            {
                "type": "room_initial",
                "time": "A",
                "question": f"In which room is the {obj} initially?",
                "answer": room_a,
                "support_node_idx": support_a_idx,
            }
        )
    if room_b:
        qa.append(
            {
                "type": "room_after_move",
                "time": "B",
                "question": f"In which room is the {obj} after it was moved?",
                "answer": room_b,
                "support_node_idx": support_b_idx,
            }
        )

    # 4) Multi-hop / relational question, if we have an edge for a support.
    # Prefer relations with memory snapshots (snapshot_step/frame) when present.
    def pick_relational(idx: int) -> Optional[Tuple[int, str, Dict[str, Any]]]:
        neigh = _neighbors_for_node(sg, idx)
        if not neigh:
            return None
        with_snap = [t for t in neigh if ("snapshot_b64" in t[2] or t[2].get("snapshot_step") is not None)]
        pool = with_snap or neigh
        return rng.choice(pool)

    rel_a = pick_relational(support_a_idx)
    if rel_a is not None:
        other_idx, rel, e = rel_a
        other_cap = cap(other_idx)
        # Use a simple spatial reference (edge relations are full sentences, not prepositions)
        q = f"The {a_cap} is near the {other_cap}. Is the {obj} on this {a_cap}?"
        qa.append(
            {
                "type": "relational_support",
                "time": "A",
                "question": q,
                "answer": "yes",
                "support_node_idx": support_a_idx,
                "relation": rel,
                "other_node_idx": other_idx,
                "snapshot_step": e.get("snapshot_step"),
                "snapshot_frame_idx": e.get("snapshot_frame_idx"),
            }
        )
        qa.append(
            {
                "type": "relational_support",
                "time": "B",
                "question": q,
                "answer": "no" if support_b_idx != support_a_idx else "yes",
                "support_node_idx": support_a_idx,
                "relation": rel,
                "other_node_idx": other_idx,
            }
        )

    return qa[:max_pairs]
