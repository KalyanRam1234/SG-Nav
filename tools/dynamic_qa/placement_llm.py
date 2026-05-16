"""
LLM-based placement suggestion for DynamicQA object insertion.

This module adds intelligent placement selection by prompting an LLM to analyze
the scene graph and suggest the 2 best locations for placing an object.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .llm_client import LLMConfig, chat
from .location_selector import SupportChoice
from .sg_cache import (
    SerializedSceneGraph,
    iter_nodes,
    node_caption,
    node_centroid_world,
    node_room_caption,
)


def _build_scene_context(
    sg: SerializedSceneGraph,
    object_category: str,
    max_nodes_in_prompt: int = 30,
) -> Tuple[str, Dict[str, int]]:
    """Build a text description of the scene graph for LLM analysis.
    
    Returns:
        (context_text, node_idx_to_id_map)
    """
    nodes = list(iter_nodes(sg))
    
    # Create a mapping of actual node indices to display IDs (for cleaner prompt)
    node_idx_to_id: Dict[str, int] = {}
    id_counter = 0
    
    # Prioritize nodes by category relevance for placement
    nodes_with_priority = []
    for i, node in enumerate(nodes):
        cap = node_caption(node).lower()
        priority = 0
        
        # Higher priority for support-type objects
        support_types = ["table", "counter", "desk", "bed", "sofa", "shelf", "cabinet", "sink"]
        if any(st in cap for st in support_types):
            priority = 100
        # Lower priority for decorative items
        elif any(cat in cap for cat in ["plant", "picture", "lamp", "rug", "curtain"]):
            priority = -50
        
        nodes_with_priority.append((i, node, priority))
    
    # Sort by priority (descending) and select top N
    nodes_with_priority.sort(key=lambda x: x[2], reverse=True)
    selected_nodes = nodes_with_priority[:max_nodes_in_prompt]
    
    lines = [
        f"SCENE CONTEXT:",
        f"Target object to place: {object_category}",
        f"",
        "Objects in scene:",
        "",
    ]
    
    for actual_idx, node, _ in selected_nodes:
        display_id = id_counter
        node_idx_to_id[str(actual_idx)] = display_id
        id_counter += 1
        
        cap = node_caption(node)
        room = node_room_caption(sg, node)
        room_str = f" (in {room})" if room else ""
        
        centroid = node_centroid_world(node)
        pos_str = f"at [{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}]" if centroid is not None else "at unknown position"
        
        lines.append(f"[ID {display_id}] {cap}{room_str} {pos_str}")
    
    return "\n".join(lines), node_idx_to_id


def _build_placement_prompt(
    scene_context: str,
    object_category: str,
) -> str:
    """Build the LLM prompt for placement suggestion.
    
    The prompt should elicit:
    1. Two distinct, sensible placement locations
    2. Reasoning about why each location is good
    3. JSON response with the selected node IDs
    """
    
    prompt = f"""You are an expert in spatial reasoning and 3D scene understanding, helping to place objects realistically in indoor scenes.

{scene_context}

TASK: Suggest the 2 BEST locations to place a "{object_category}" in this scene.

Requirements:
- Placement location should be on a horizontal surface (table, counter, desk, bed, shelf, etc.)
- Choose 2 DIFFERENT surfaces to maximize scene variation
- Prefer surfaces in different rooms if possible
- Choose surfaces that make semantic sense (e.g., phone on desk, cup on counter)
- Avoid placing items on top of other small objects or in illogical locations

For each suggested placement:
1. Identify which object (by ID) would be a good support surface
2. Briefly explain why that location makes sense (1-2 sentences)

IMPORTANT: You MUST respond with ONLY valid JSON (no markdown, no extra text).

JSON FORMAT (required):
{{
    "placements": [
        {{
            "id": <node_id>,
            "support_object": "<object name>",
            "reasoning": "<1-2 sentence explanation>"
        }},
        {{
            "id": <node_id>,
            "support_object": "<object name>",
            "reasoning": "<1-2 sentence explanation>"
        }}
    ]
}}

Respond with ONLY the JSON object, nothing else."""
    
    return prompt


def suggest_placements_llm(
    sg: SerializedSceneGraph,
    object_category: str,
    llm_cfg: Optional[LLMConfig] = None,
    verbose: bool = False,
) -> List[SupportChoice]:
    """Query an LLM to suggest the 2 best placement locations.
    
    Args:
        sg: Scene graph dictionary
        object_category: Category of object to place (e.g., "cup")
        llm_cfg: LLM configuration (uses defaults if None)
        verbose: Print debug information
        
    Returns:
        List of 2 SupportChoice objects (node_idx, caption, room_idx)
        
    Raises:
        ValueError: If LLM fails to return valid JSON or insufficient suggestions
    """
    
    llm_cfg = llm_cfg or LLMConfig()
    
    # Build scene context
    scene_context, node_idx_to_id = _build_scene_context(sg, object_category)
    
    if verbose:
        print(f"[Placement LLM] Scene context built with {len(node_idx_to_id)} nodes")
        print(f"[Placement LLM] Querying LLM model: {llm_cfg.model}")
    
    # Build prompt
    prompt = _build_placement_prompt(scene_context, object_category)
    
    # Call LLM
    try:
        response = chat(prompt, llm_cfg, json_mode=True)
    except Exception as e:
        raise ValueError(f"LLM placement suggestion failed: {e}")
    
    if verbose:
        print(f"[Placement LLM] Raw response: {response[:200]}...")
    
    # Parse JSON response
    try:
        # Try to extract JSON from response (in case there's extra text despite json_mode)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {response[:300]}") from e
    
    placements = data.get("placements", [])
    if len(placements) < 2:
        raise ValueError(f"LLM returned only {len(placements)} placement(s), need 2")
    
    # Map display IDs back to actual node indices
    nodes = list(iter_nodes(sg))
    reverse_map: Dict[int, int] = {v: int(k) for k, v in node_idx_to_id.items()}
    
    result: List[SupportChoice] = []
    for placement in placements[:2]:  # Take first 2
        display_id = placement.get("id")
        if display_id not in reverse_map:
            raise ValueError(f"LLM returned invalid node ID: {display_id}")
        
        actual_idx = reverse_map[display_id]
        node = nodes[actual_idx]
        caption = node_caption(node)
        room_idx = node.get("room_idx")
        
        choice = SupportChoice(
            node_idx=actual_idx,
            caption=caption,
            room_idx=room_idx,
        )
        result.append(choice)
        
        if verbose:
            reason = placement.get("reasoning", "")
            print(f"[Placement LLM]   Placement {len(result)}: '{caption}' (idx={actual_idx})")
            print(f"[Placement LLM]     Reasoning: {reason}")
    
    return result
