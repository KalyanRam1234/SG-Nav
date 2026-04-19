from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from .llm_client import LLMConfig, chat
from .sg_cache import SerializedSceneGraph, build_support_graph_text, iter_edges


def _extract_json(text: str) -> str:
    """Extract the first JSON array or object from model output."""
    m = re.search(r"(\[.*\])", text, flags=re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m:
        return m.group(1)
    return text.strip()


def _neighbors(sg: SerializedSceneGraph, seed_idxs: Sequence[int], max_nodes: int = 60) -> List[int]:
    nodes = sg.get("nodes") or []
    keep = set(int(i) for i in seed_idxs if 0 <= int(i) < len(nodes))

    # 1-hop neighborhood via edges
    for e in iter_edges(sg):
        i = e.get("node1_idx")
        j = e.get("node2_idx")
        if i in keep and j is not None:
            keep.add(int(j))
        if j in keep and i is not None:
            keep.add(int(i))
        if len(keep) >= max_nodes:
            break

    return sorted(keep)


def _model_supports_images(model: str) -> bool:
    m = (model or "").lower()
    # Heuristic: common Ollama VLM name patterns.
    if "vision" in m or "llava" in m or "moondream" in m or "minicpm" in m:
        return True
    if "qwen" in m and "vl" in m:
        return True
    return False


def _should_use_snapshots(use_snapshots: str, model: str) -> bool:
    u = (use_snapshots or "auto").lower().strip()
    if u in ("on", "true", "1", "yes"):
        return True
    if u in ("off", "false", "0", "no"):
        return False
    return _model_supports_images(model)


def _collect_snapshot_evidence(
    sg: SerializedSceneGraph,
    node_idxs: Sequence[int],
    *,
    support_idxs: Sequence[int],
    max_edges: int,
    max_images: int,
) -> tuple[list[str], list[str]]:
    """Collect snapshot_b64 images from nodes/edges touching node_idxs.

    Returns:
      - images: list of base64-encoded JPEGs (no data-uri prefix)
      - lines: human-readable mapping lines describing each image
    """

    nodes = sg.get("nodes") or []
    idx_set = set(int(i) for i in node_idxs)
    support_set = set(int(i) for i in support_idxs)

    edges = sg.get("edges") or []
    candidates = []

    used = 0
    for e in edges:
        if used >= max_edges:
            break
        if not isinstance(e, dict):
            continue
        i = e.get("node1_idx")
        j = e.get("node2_idx")
        if i not in idx_set and j not in idx_set:
            continue
        used += 1

        b64 = e.get("snapshot_b64")
        if not isinstance(b64, str) or not b64:
            continue

        score = 0
        if i in support_set or j in support_set:
            score += 100000
        step = e.get("snapshot_step")
        if isinstance(step, int):
            score += step

        candidates.append((score, e))

    candidates.sort(key=lambda t: t[0], reverse=True)

    images: list[str] = []
    lines: list[str] = []
    seen = set()

    def cap(idx: int) -> str:
        if 0 <= idx < len(nodes):
            return str(nodes[idx].get("caption"))
        return f"node_{idx}"

    # Node snapshots (rare, but support if present)
    for idx in node_idxs:
        if len(images) >= max_images:
            break
        if not (0 <= int(idx) < len(nodes)):
            continue
        n = nodes[int(idx)]
        if not isinstance(n, dict):
            continue
        b64 = n.get("snapshot_b64")
        if not isinstance(b64, str) or not b64:
            obj = n.get("object")
            if isinstance(obj, dict):
                b64 = obj.get("snapshot_b64")
        if not isinstance(b64, str) or not b64:
            continue
        if b64 in seen:
            continue
        seen.add(b64)
        img_idx = len(images)
        lines.append(f"- img{img_idx}: node ({int(idx)}) {cap(int(idx))} (node snapshot)")
        images.append(b64)

    for _score, e in candidates:
        if len(images) >= max_images:
            break
        b64 = e.get("snapshot_b64")
        if not isinstance(b64, str) or not b64:
            continue
        if b64 in seen:
            continue
        seen.add(b64)

        i = int(e.get("node1_idx", -1))
        j = int(e.get("node2_idx", -1))
        rel = str(e.get("relation", ""))
        meta = []
        if e.get("snapshot_step") is not None:
            meta.append(f"step={e.get('snapshot_step')}")
        if e.get("snapshot_frame_idx") is not None:
            meta.append(f"frame={e.get('snapshot_frame_idx')}")
        meta_s = "" if not meta else " (" + ", ".join(meta) + ")"

        img_idx = len(images)
        lines.append(f"- img{img_idx}: edge ({i}) {cap(i)} -- {rel} --> ({j}) {cap(j)}{meta_s}")
        images.append(b64)

    return images, lines


def generate_qa_pairs_llm(
    sg: SerializedSceneGraph,
    inserted_object: str,
    support_a_idx: int,
    support_b_idx: int,
    qa_count: int = 20,
    seed: int = 0,
    llm_cfg: Optional[LLMConfig] = None,
    max_retries: int = 2,
    verbose: bool = False,
    *,
    use_snapshots: str = "auto",
    max_images: int = 6,
) -> List[Dict[str, Any]]:
    """LLM-based QA generation using serialized SG (nodes/rooms/edges + snapshot metadata)."""

    nodes = sg.get("nodes") or []

    def cap(i: int) -> str:
        if 0 <= i < len(nodes):
            return str(nodes[i].get("caption"))
        return f"node_{i}"

    a_cap = cap(int(support_a_idx))
    b_cap = cap(int(support_b_idx))

    llm_cfg = llm_cfg or LLMConfig()

    focus = _neighbors(sg, [support_a_idx, support_b_idx])
    max_edges = 240
    subgraph_text = build_support_graph_text(sg, focus, max_edges=max_edges, include_snapshot_meta=True)

    images: Optional[List[str]] = None
    evidence_lines: List[str] = []
    if _should_use_snapshots(use_snapshots, str(llm_cfg.model)):
        imgs, lines = _collect_snapshot_evidence(
            sg,
            focus,
            support_idxs=[support_a_idx, support_b_idx],
            max_edges=max_edges,
            max_images=int(max_images),
        )
        images = imgs or None
        evidence_lines = lines

    if verbose:
        total_nodes = len(nodes)
        total_edges = len(sg.get("edges") or [])
        print(
            f"[DynamicQA][LLM] Subgraph focus nodes={len(focus)} (scene nodes={total_nodes}, edges={total_edges}) "
            f"supports=({support_a_idx}, {support_b_idx})"
        )
        if images:
            print(f"[DynamicQA][LLM] Attaching {len(images)} snapshot image(s)")

    def build_prompt(evidence_lines_cur: List[str]) -> str:
        evidence_text = "" if not evidence_lines_cur else "\n".join(evidence_lines_cur)
        return (
            f"""
You are generating a Dynamic Scene QA dataset.

SCENARIO
- There is an inserted dynamic object: "{inserted_object}".
- Time A: the {inserted_object} is placed ON/AT the support object "{a_cap}" (node {support_a_idx}).
- Time B: the {inserted_object} is moved ON/AT the support object "{b_cap}" (node {support_b_idx}).
- Everything else in the scene graph is unchanged.

SCENE GRAPH (subgraph around supports)
{subgraph_text}

EVIDENCE IMAGES
- You may be given 0 or more images (img0..imgN-1) that correspond to memory snapshots for some relation edges in the graph.
- Each image is a thumbnail captured when establishing that edge.
- If images are provided, use them as additional grounding for relationships and object appearance.
- Do NOT mention image indices (img0, img1, ...) in your questions or answers.
{evidence_text}

TASK
Create {qa_count} diverse QA pairs that require understanding of relationships in the scene graph.
Constraints:
- Avoid trivial questions like only "Is X present?".
- At least half the questions must be relational/multi-hop (use edges/relations like next-to/on/inside/etc.).
- Include temporal reasoning across A vs B (e.g., "before/after", "still", "moved", "used to be").
- Prefer questions that can be answered from the graph + the swap description.
- Answers must be short strings ("yes"/"no", a node caption, a room caption, or a small phrase).

OUTPUT FORMAT
Return ONLY a JSON array. Each element must be an object with keys:
- question: string
- answer: string
- time: "A" | "B" | "AB"  (AB means requires comparing A vs B)
- type: short tag (e.g., "temporal", "relational", "room", "multi_hop", "comparison")
- grounding_node_idxs: array of integers (include support nodes and any referenced nodes)

Do not include markdown fences.
"""
            .strip()
        )

    images_cur = images
    evidence_lines_cur = list(evidence_lines)
    prompt = build_prompt(evidence_lines_cur)

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            if verbose:
                print(f"[DynamicQA][LLM] Calling model (attempt {attempt + 1}/{max_retries + 1})")
            raw = chat(prompt, llm_cfg, json_mode=True, images=images_cur)
            if verbose:
                snippet = raw.replace("\n", " ")
                if len(snippet) > 240:
                    snippet = snippet[:240] + "..."
                print(f"[DynamicQA][LLM] Raw response (trunc): {snippet}")
            payload = _extract_json(raw)
            data = json.loads(payload)
            if not isinstance(data, list):
                raise ValueError("LLM output is not a JSON array")

            out: List[Dict[str, Any]] = []
            kept = 0
            for item in data:
                if not isinstance(item, dict):
                    continue
                q = item.get("question")
                a = item.get("answer")
                t = item.get("time")
                if not isinstance(q, str) or not isinstance(a, str) or t not in ("A", "B", "AB"):
                    continue
                g = item.get("grounding_node_idxs")
                if not isinstance(g, list):
                    g = [support_a_idx, support_b_idx]
                out.append(
                    {
                        "question": q.strip(),
                        "answer": a.strip(),
                        "time": t,
                        "type": str(item.get("type", "llm")),
                        "grounding_node_idxs": [int(x) for x in g if isinstance(x, (int, float, str))],
                        "seed": seed,
                    }
                )
                kept += 1

            if verbose:
                print(f"[DynamicQA][LLM] Parsed {len(data)} item(s), accepted {kept}, returning {min(len(out), qa_count)}")

            # Keep deterministic size cap
            return out[:qa_count]
        except Exception as e:
            last_err = e
            msg = str(e).lower()

            # If the backend/model rejects images entirely, retry without them.
            if images_cur and ("images" in msg and ("not supported" in msg or "unsupported" in msg)):
                images_cur = None
                evidence_lines_cur = []
                prompt = build_prompt(evidence_lines_cur)
                if verbose:
                    print("[DynamicQA][LLM] Model rejected images; retrying with snapshots disabled")
                continue

            if verbose:
                print(f"[DynamicQA][LLM] Attempt failed: {e}")

    raise RuntimeError(
        "Failed to generate LLM QA pairs after retries: "
        f"{last_err} (try --llm_max_images 1 or --llm_use_snapshots off)"
    )
