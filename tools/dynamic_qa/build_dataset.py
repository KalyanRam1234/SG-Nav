from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .location_selector import select_support_pair, get_all_support_candidates, SupportChoice
from .qa_generator import generate_qa_pairs
from .qa_llm import generate_qa_pairs_llm
from .llm_client import LLMConfig
from .scene_generator import propose_drop_start, settle_object_with_physics, check_navmesh_proximity
from .sg_cache import load_global_sg_pkl, node_centroid_world, iter_nodes


def _write_records(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write JSONL (default) or JSON depending on file extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".json") and not str(path).endswith(".jsonl"):
        with path.open("w") as f:
            json.dump(records, f, indent=2)
        return

    # Default: JSONL
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def build_record(
    sg_pkl: Path,
    inserted_object_category: str,
    object_template_handle: str,
    variant_a: str = "A",
    variant_b: str = "B",
    run_physics: bool = False,
    scene_handle: Optional[str] = None,
    seed: int = 0,
    qa_mode: str = "templated",
    qa_count: int = 20,
    llm_cfg: Optional[LLMConfig] = None,
    llm_use_snapshots: str = "auto",
    llm_max_images: int = 6,
    max_navmesh_dist: float = 2.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Build a Krutik-style DynamicQA record.

    Output keys (per record):
      - object_category
      - scene_handle
      - template_handle
      - QA_pairs (question/answer)
      - validPlacements (list of {position, rotation})

    Note: we map variant A/B placements to validPlacements[0]/[1].
    """

    save = load_global_sg_pkl(sg_pkl)
    sg = save.scenegraph

    # Resolve scene handle (older pkls may not include it)
    scene_handle = scene_handle or save.scene_id
    if scene_handle is None:
        raise ValueError(
            "scene_handle is required (this SG pkl does not embed it). "
            "Pass --scene_handle /abs/path/to/<scene>.glb"
        )

    scene_handle_abs = str(Path(scene_handle).expanduser().resolve())
    template_handle_abs = str(Path(object_template_handle).expanduser().resolve())

    if verbose:
        print(f"[DynamicQA] Building record from sg_pkl={sg_pkl}")
        print(f"[DynamicQA] scene_handle={scene_handle_abs}")
        print(f"[DynamicQA] object_category={inserted_object_category}")
        print(f"[DynamicQA] template_handle={template_handle_abs}")
        print(f"[DynamicQA] max_navmesh_dist={max_navmesh_dist}m")

    # --- Navmesh-aware support selection ---
    nodes = list(iter_nodes(sg))
    all_candidates = get_all_support_candidates(sg, inserted_object_category)
    if len(all_candidates) < 2:
        raise ValueError(
            f"Not enough support candidates for '{inserted_object_category}'. "
            f"Found {len(all_candidates)} candidates."
        )

    # Pre-filter candidates by navmesh proximity of their centroid
    rng = np.random.default_rng(seed)
    centroids = []
    for c in all_candidates:
        cent = node_centroid_world(nodes[c.node_idx])
        if cent is not None:
            centroids.append(cent.tolist())
        else:
            centroids.append([0.0, 0.0, 0.0])

    nav_dists = check_navmesh_proximity(scene_handle_abs, centroids)

    valid_candidates: List[SupportChoice] = []
    for c, d in zip(all_candidates, nav_dists):
        if verbose:
            print(f"[DynamicQA]   candidate '{c.caption}' (idx={c.node_idx}, room={c.room_idx}) navmesh_dist={d:.2f}m", end="")
        if d <= max_navmesh_dist:
            valid_candidates.append(c)
            if verbose:
                print(" ✓")
        else:
            if verbose:
                print(f" ✗ (>{max_navmesh_dist}m)")

    if len(valid_candidates) < 2:
        raise ValueError(
            f"Not enough navmesh-reachable support candidates for '{inserted_object_category}'. "
            f"Found {len(valid_candidates)} valid (within {max_navmesh_dist}m of navmesh) "
            f"out of {len(all_candidates)} total. "
            f"Try increasing --max_navmesh_dist."
        )

    # Pick A/B: prefer distinct rooms
    a = valid_candidates[0]
    b = None
    if a.room_idx is not None:
        for c in valid_candidates[1:]:
            if c.room_idx is not None and c.room_idx != a.room_idx:
                b = c
                break
    if b is None:
        b = valid_candidates[1]

    if verbose:
        print(f"[DynamicQA] Selected supports: A=({a.node_idx}) '{a.caption}', B=({b.node_idx}) '{b.caption}'")

    support_a = nodes[a.node_idx]
    support_b = nodes[b.node_idx]

    placement_a: Dict[str, Any] = {
        "support_node_idx": a.node_idx,
        "support_caption": a.caption,
        "start_position": propose_drop_start(support_a, rng=rng),
    }
    placement_b: Dict[str, Any] = {
        "support_node_idx": b.node_idx,
        "support_caption": b.caption,
        "start_position": propose_drop_start(support_b, rng=rng),
    }

    if verbose:
        print(f"[DynamicQA] Proposed drop starts: A={placement_a['start_position']}, B={placement_b['start_position']}")

    # Always settle — uses Bullet physics if available, geometric fallback otherwise
    if verbose:
        print("[DynamicQA] Running object settling...")
    settled_a = settle_object_with_physics(scene_handle_abs, template_handle_abs, placement_a["start_position"])
    settled_b = settle_object_with_physics(scene_handle_abs, template_handle_abs, placement_b["start_position"])
    placement_a["position"] = settled_a.position
    placement_a["rotation_quat_xyzw"] = settled_a.rotation_quat_xyzw
    placement_a["settle_steps"] = settled_a.steps
    placement_a["settled_with_physics"] = settled_a.settled
    placement_a["navmesh_dist"] = settled_a.navmesh_dist
    placement_b["position"] = settled_b.position
    placement_b["rotation_quat_xyzw"] = settled_b.rotation_quat_xyzw
    placement_b["settle_steps"] = settled_b.steps
    placement_b["settled_with_physics"] = settled_b.settled
    placement_b["navmesh_dist"] = settled_b.navmesh_dist
    if verbose:
        mode = "Bullet physics" if settled_a.settled else "geometric"
        print(f"[DynamicQA] Settled A ({mode}) pos={settled_a.position} navmesh_dist={settled_a.navmesh_dist:.2f}m")
        mode = "Bullet physics" if settled_b.settled else "geometric"
        print(f"[DynamicQA] Settled B ({mode}) pos={settled_b.position} navmesh_dist={settled_b.navmesh_dist:.2f}m")

    if qa_mode == "llm":
        llm_cfg = llm_cfg or LLMConfig()
        if verbose:
            print(
                "[DynamicQA] Generating QA via LLM "
                f"(backend={llm_cfg.backend}, model={llm_cfg.model}, temp={llm_cfg.temperature}, count={qa_count})"
            )
        qa_pairs = generate_qa_pairs_llm(
            sg,
            inserted_object=inserted_object_category,
            support_a_idx=a.node_idx,
            support_b_idx=b.node_idx,
            qa_count=qa_count,
            seed=seed,
            llm_cfg=llm_cfg,
            verbose=verbose,
            use_snapshots=str(llm_use_snapshots),
            max_images=int(llm_max_images),
        )
        if verbose:
            print(f"[DynamicQA] LLM QA generation returned {len(qa_pairs)} pair(s)")
            if qa_pairs:
                q0 = qa_pairs[0]
                if isinstance(q0, dict) and 'question' in q0:
                    print(f"[DynamicQA] Sample QA[0]: {q0.get('question')} -> {q0.get('answer')}")
    else:
        if verbose:
            print(f"[DynamicQA] Generating QA via templated mode (max_pairs={qa_count})")
        qa_pairs = generate_qa_pairs(
            sg,
            inserted_object=inserted_object_category,
            support_a_idx=a.node_idx,
            support_b_idx=b.node_idx,
            seed=seed,
            max_pairs=qa_count,
        )

    QA_pairs = []
    for q in qa_pairs:
        if not isinstance(q, dict):
            continue
        qq = q.get("question")
        aa = q.get("answer")
        if isinstance(qq, str) and isinstance(aa, str):
            QA_pairs.append({"question": qq, "answer": aa})

    def as_valid(p: Dict[str, Any]) -> Dict[str, Any]:
        pos = p.get("position") or p.get("start_position")
        rot = p.get("rotation_quat_xyzw")
        if rot is None:
            rot = [0.0, 0.0, 0.0, 1.0]
        return {"position": pos, "rotation": rot}

    record: Dict[str, Any] = {
        "object_category": inserted_object_category,
        "scene_handle": scene_handle_abs,
        "template_handle": template_handle_abs,
        "QA_pairs": QA_pairs,
        "validPlacements": [as_valid(placement_a), as_valid(placement_b)],
    }
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Dynamic Scene QA manifest from saved global_sg_*.pkl files")
    ap.add_argument("--sg_pkl", required=True, nargs='+', help="One or more paths to global_sg_*.pkl")
    ap.add_argument("--scene_handle", default=None, help="Scene handle (.glb). Required if SG pkl doesn't embed scene_id (common for older pkls).")
    ap.add_argument("--cad_category", required=True, help="Inserted CAD object category (e.g., phone)")
    ap.add_argument(
        "--cad_template_handle",
        required=True,
        help="Habitat-Sim object template config (e.g., *.object_config.json)",
    )
    ap.add_argument("--output", required=True, help="Output manifest path (.jsonl or .json)")
    ap.add_argument("--run_physics", action="store_true", help="(Legacy) Physics settling always runs; uses Bullet if available, geometric fallback otherwise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of records")

    ap.add_argument("--qa_mode", choices=["templated", "llm"], default="templated")
    ap.add_argument("--qa_count", type=int, default=20)
    ap.add_argument("--llm_model", type=str, default="llama3.2-vision:latest")
    ap.add_argument("--llm_backend", choices=["ollama", "http"], default="ollama")
    ap.add_argument("--llm_url", type=str, default="http://localhost:11434/api/chat")
    ap.add_argument("--llm_temperature", type=float, default=0.2)
    ap.add_argument(
        "--llm_timeout_s",
        type=float,
        default=600.0,
        help="Timeout (seconds) for a single LLM call. Prevents indefinite hangs.",
    )
    ap.add_argument(
        "--llm_num_predict",
        type=int,
        default=1536,
        help="Max tokens to generate (Ollama num_predict). Lower = faster, too low may truncate JSON.",
    )
    ap.add_argument(
        "--llm_keep_alive",
        type=str,
        default=None,
        help="Keep model loaded between calls (e.g. '10m'). Useful when building many records.",
    )
    ap.add_argument(
        "--llm_use_snapshots",
        choices=["auto", "on", "off"],
        default="auto",
        help="Attach SG memory snapshots (edge snapshot_b64) as images to the LLM call. 'auto' enables for vision models.",
    )
    ap.add_argument(
        "--llm_max_images",
        type=int,
        default=6,
        help="Max snapshot images to attach per LLM call (kept small for speed).",
    )
    ap.add_argument(
        "--llm_progress",
        action="store_true",
        help="Print progress dots while waiting for the LLM (requires streaming).",
    )
    ap.add_argument("--verbose", action="store_true", help="Print debug info for dataset generation")
    ap.add_argument("--max_navmesh_dist", type=float, default=2.0,
                    help="Max distance (m) from placement centroid to nearest navigable point. "
                         "Supports beyond this are rejected. Default: 2.0")

    args = ap.parse_args()

    records: List[Dict[str, Any]] = []
    sg_paths = [Path(p) for p in args.sg_pkl]
    if args.limit is not None:
        sg_paths = sg_paths[: int(args.limit)]

    llm_cfg = LLMConfig(
        backend=str(args.llm_backend),
        model=str(args.llm_model),
        temperature=float(args.llm_temperature),
        url=str(args.llm_url),
        timeout_s=float(args.llm_timeout_s),
        num_predict=int(args.llm_num_predict),
        keep_alive=(str(args.llm_keep_alive) if args.llm_keep_alive is not None else None),
        stream=bool(args.llm_progress),
        show_progress=bool(args.llm_progress),
    )

    for i, p in enumerate(sg_paths):
        rec = build_record(
            sg_pkl=p,
            inserted_object_category=args.cad_category,
            object_template_handle=args.cad_template_handle,
            run_physics=args.run_physics,
            scene_handle=args.scene_handle,
            seed=int(args.seed) + i,
            qa_mode=str(args.qa_mode),
            qa_count=int(args.qa_count),
            llm_cfg=llm_cfg,
            llm_use_snapshots=str(args.llm_use_snapshots),
            llm_max_images=int(args.llm_max_images),
            max_navmesh_dist=float(args.max_navmesh_dist),
            verbose=bool(args.verbose),
        )
        records.append(rec)

    _write_records(Path(args.output), records)
    print(f"Wrote {len(records)} record(s) to {args.output}")


if __name__ == "__main__":
    main()
