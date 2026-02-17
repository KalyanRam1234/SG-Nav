# LLM-Guided Stuck Recovery — Implementation Plan

## Problem

The robot frequently gets stuck during navigation. The current recovery logic in `SG_Nav.py` (lines 828–899) is primitive: it tries FBE frontiers, random goals, and forced turn/forward sequences in a fixed pattern. This often fails to escape complex stuck scenarios (e.g., corners, narrow passages, oscillating between two positions).

## Proposed Solution

Add an **LLM-guided escape mode** that, when the robot is stuck:

1. Serializes the current scene graph, spatial maps, and agent state into a structured prompt
2. Sends the current camera image + prompt to the **VLM** (`llama3.2-vision` via Ollama)
3. Parses the VLM's response into a sequence of named actions (`move_forward`, `turn_left`, `turn_right`)
4. Executes those actions sequentially on subsequent `act()` calls, bypassing normal planning
5. After the escape sequence completes, resumes normal `act()` flow

**Gated behind `--llm_escape` flag.** When disabled, the existing primitive recovery logic runs as before.

## Design Decisions

| Decision | Choice |
|---|---|
| Max actions per escape plan | 15 |
| Max LLM escape attempts before fallback | 2 |
| Model | VLM (`llama3.2-vision`) with camera image + scene graph + spatial maps |
| Action format in prompt | Named: `move_forward`, `turn_left`, `turn_right` |

## Architecture / Flow

```
act() is called
│
├─ IF self.executing_escape AND self.escape_action_queue is non-empty:
│   └─ Pop next action from queue → return it (skip ALL normal planning)
│   └─ If queue becomes empty → set self.executing_escape = False
│
├─ ... normal act() logic ...
│
├─ Stuck detected (line 833 condition):
│   ├─ IF self.args.llm_escape AND self.escape_attempts < 2:
│   │   └─ Call self.llm_plan_escape(observations, traversible, cur_start)
│   │       ├─ _serialize_scene_graph_for_vlm()  → structured text
│   │       ├─ _get_spatial_map_summary()         → free/collision/room stats
│   │       ├─ _build_escape_prompt()             → full VLM prompt
│   │       ├─ scenegraph.get_vlm_response()      → VLM call with camera image
│   │       ├─ _parse_escape_actions()            → list of int actions
│   │       └─ Store in self.escape_action_queue, set self.executing_escape = True
│   │
│   └─ ELSE (flag off OR attempts exhausted):
│       └─ Existing primitive recovery logic (FBE, random goals, forced turns)
```

---

## Detailed Code Changes

### File 1: `SG_Nav.py`

#### Change 1.1 — Add `--llm_escape` CLI flag
**Location:** Lines 1469–1479 (argparse block at bottom of file)

Add after the `--split_r` argument:
```python
parser.add_argument(
    "--llm_escape", action='store_true',
    help="Enable LLM-guided escape when robot is stuck"
)
```

#### Change 1.2 — Add escape state variables to `__init__`
**Location:** After line 165 (`self.detected_objects_extended = set()`)

Add:
```python
# LLM escape mode state
self.escape_action_queue = []
self.executing_escape = False
self.escape_attempts = 0
self.max_escape_attempts = 2
self.max_escape_actions = 15
self.action_history = []  # Rolling buffer of recent actions for context
```

#### Change 1.3 — Reset escape state in `reset()` method
**Location:** After line 277 (`self.detected_objects_extended = set()`)

Add:
```python
self.escape_action_queue = []
self.executing_escape = False
self.escape_attempts = 0
self.action_history = []
```

#### Change 1.4 — Reset escape state in `reset_local_scenegraph()`
**Location:** After line 340 (`self.text_edge = ''`)

Add:
```python
self.escape_action_queue = []
self.executing_escape = False
self.escape_attempts = 0
```

#### Change 1.5 — Track action history
**Location:** Line 911 (`self.prev_action = number_action`)

Add after:
```python
self.action_history.append(number_action)
if len(self.action_history) > 20:
    self.action_history = self.action_history[-20:]
```

#### Change 1.6 — Add escape queue execution at top of `act()`
**Location:** After line 564 (`self.exploration_total_steps += 1`), before the "Determine current target object" block

Add a new block:
```python
# LLM escape mode: execute queued escape actions
if self.executing_escape and self.escape_action_queue:
    escape_action = self.escape_action_queue.pop(0)
    print(f"[Act] LLM ESCAPE: Executing action {escape_action} "
          f"({len(self.escape_action_queue)} remaining)")
    
    if not self.escape_action_queue:
        print(f"[Act] LLM ESCAPE: Sequence complete, resuming normal navigation")
        self.executing_escape = False
    
    self.prev_action = escape_action
    self.action_history.append(escape_action)
    if len(self.action_history) > 20:
        self.action_history = self.action_history[-20:]
    self.navigate_steps += 1
    return {"action": escape_action}
```

#### Change 1.7 — New method: `_serialize_scene_graph_for_vlm()`
**Location:** Add as a new method in the `SG_Nav_Agent` class (after `print_metrics()`, around line 535)

This method builds a hierarchical text representation of the scene state:

```python
def _serialize_scene_graph_for_vlm(self):
    """Serialize scene graph + spatial state for VLM escape prompt."""
    lines = []
    
    # 1. Agent state
    pose = self.full_pose.cpu().numpy()
    lines.append(f"=== AGENT STATE ===")
    lines.append(f"Position: ({pose[0]:.2f}, {pose[1]:.2f})")
    lines.append(f"Orientation: {pose[2]:.1f} degrees")
    lines.append(f"Target object: {self.obj_goal}")
    lines.append(f"Steps taken: {self.total_steps}")
    lines.append(f"Steps without moving: {self.not_move_steps}")
    
    # 2. Recent action history
    action_names = {0: 'stop', 1: 'move_forward', 2: 'turn_left', 3: 'turn_right', 6: 'panoramic'}
    recent = [action_names.get(a, str(a)) for a in self.action_history[-10:]]
    lines.append(f"Recent actions: {', '.join(recent)}")
    
    # 3. Global scene graph (hierarchical: rooms → objects → edges)
    lines.append(f"\n=== SCENE GRAPH ===")
    
    # Room nodes
    if hasattr(self.global_scenegraph, 'room_nodes') and self.global_scenegraph.room_nodes:
        room_names = [r.caption for r in self.global_scenegraph.room_nodes if r.nodes]
        if room_names:
            lines.append(f"Rooms with objects: {', '.join(room_names)}")
    
    # Object nodes
    global_objects = sorted(set(node.caption for node in self.global_scenegraph.nodes))
    local_objects = sorted(set(node.caption for node in self.scenegraph.nodes))
    lines.append(f"Objects seen globally: {', '.join(global_objects) if global_objects else 'none'}")
    lines.append(f"Objects seen locally: {', '.join(local_objects) if local_objects else 'none'}")
    
    # Edges (relationships)
    edges = self.global_scenegraph.get_edges()
    if edges:
        edge_texts = [f"{e.node1.caption} {e.relation} {e.node2.caption}" for e in edges[:20]]
        lines.append(f"Spatial relationships: {'; '.join(edge_texts)}")
    
    # 4. Spatial map summary
    lines.append(f"\n=== SPATIAL MAP ===")
    lines.append(self._get_spatial_map_summary())
    
    return '\n'.join(lines)
```

#### Change 1.8 — New method: `_get_spatial_map_summary()`
**Location:** Add right after `_serialize_scene_graph_for_vlm()`

Summarizes the spatial maps (free space, collisions, explored area):

```python
def _get_spatial_map_summary(self):
    """Summarize spatial maps for the VLM prompt."""
    lines = []
    
    # Free map coverage
    if hasattr(self, 'global_fbe_free_map') and self.global_fbe_free_map is not None:
        free_map = self.global_fbe_free_map[0, 0].cpu().numpy()
        total_cells = free_map.size
        free_cells = (free_map > 0).sum()
        lines.append(f"Explored area: {free_cells}/{total_cells} cells "
                     f"({100*free_cells/total_cells:.1f}%)")
    
    # Collision map
    if hasattr(self, 'global_collision_map'):
        collision_cells = (self.global_collision_map > 0).sum()
        lines.append(f"Collision cells: {collision_cells}")
    
    # Agent's local surroundings (5x5 grid around agent position)
    pose = self.full_pose.cpu().numpy()
    agent_x = int(self.map_size_cm / 10 - pose[1] * 100 / self.resolution)
    agent_y = int(self.map_size_cm / 10 + pose[0] * 100 / self.resolution)
    agent_x = max(2, min(agent_x, self.map_size - 3))
    agent_y = max(2, min(agent_y, self.map_size - 3))
    
    if hasattr(self, 'global_collision_map'):
        local_collisions = self.global_collision_map[
            agent_x-2:agent_x+3, agent_y-2:agent_y+3]
        lines.append(f"Nearby collision pattern (5x5 grid around agent):")
        for row in local_collisions:
            lines.append('  ' + ' '.join(['X' if c > 0 else '.' for c in row]))
    
    # Room detection
    if self.current_rooms:
        lines.append(f"Current room(s): {', '.join(self.current_rooms)}")
    
    return '\n'.join(lines)
```

#### Change 1.9 — New method: `_build_escape_prompt()`
**Location:** Add right after `_get_spatial_map_summary()`

```python
def _build_escape_prompt(self):
    """Build the VLM prompt for escape planning."""
    scene_context = self._serialize_scene_graph_for_vlm()
    
    prompt = f"""You are controlling a robot that is STUCK and cannot make progress toward its navigation goal.

{scene_context}

The robot's available actions are:
- move_forward: Move 0.25m in the direction the robot is facing
- turn_left: Rotate 30 degrees left
- turn_right: Rotate 30 degrees right

The attached image shows the robot's current camera view.

Based on the scene graph, spatial map, and camera view:
1. Analyze WHY the robot is stuck (wall, corner, obstacle, oscillating?)
2. Provide a sequence of actions to escape the stuck position and make progress toward finding: {self.obj_goal}

RULES:
- Output ONLY a comma-separated list of actions, max {self.max_escape_actions} actions
- Use exactly these names: move_forward, turn_left, turn_right
- Do NOT include any other text, explanation, or formatting

Example output:
turn_right, turn_right, move_forward, move_forward, turn_left, move_forward"""
    
    return prompt
```

#### Change 1.10 — New method: `_parse_escape_actions()`
**Location:** Add right after `_build_escape_prompt()`

```python
def _parse_escape_actions(self, vlm_response):
    """Parse VLM response into a list of integer actions."""
    action_map = {
        'move_forward': 1,
        'turn_left': 2,
        'turn_right': 3,
    }
    
    actions = []
    # Clean and split the response
    response_clean = vlm_response.strip().lower()
    # Remove any markdown formatting
    response_clean = response_clean.replace('`', '').replace('*', '')
    tokens = [t.strip().strip('.') for t in response_clean.split(',')]
    
    for token in tokens:
        # Try exact match first
        if token in action_map:
            actions.append(action_map[token])
        # Try fuzzy matching
        elif 'forward' in token:
            actions.append(1)
        elif 'left' in token:
            actions.append(2)
        elif 'right' in token:
            actions.append(3)
    
    # Cap at max actions
    actions = actions[:self.max_escape_actions]
    
    return actions
```

#### Change 1.11 — New method: `llm_plan_escape()`
**Location:** Add right after `_parse_escape_actions()`

```python
def llm_plan_escape(self, observations):
    """Use VLM to plan an escape sequence when the robot is stuck."""
    print(f"[LLM Escape] Planning escape (attempt {self.escape_attempts + 1}/{self.max_escape_attempts})")
    
    # Build prompt
    prompt = self._build_escape_prompt()
    print(f"[LLM Escape] Prompt:\n{prompt}")
    
    # Get VLM response with current camera image
    from PIL import Image
    rgb_image = Image.fromarray(observations["rgb"])
    response = self.scenegraph.get_vlm_response(prompt=prompt, image=rgb_image)
    print(f"[LLM Escape] VLM response: {response}")
    
    # Parse actions
    actions = self._parse_escape_actions(response)
    print(f"[LLM Escape] Parsed actions: {actions}")
    
    self.escape_attempts += 1
    
    if actions:
        self.escape_action_queue = actions
        self.executing_escape = True
        print(f"[LLM Escape] Queued {len(actions)} escape actions")
        return True
    else:
        print(f"[LLM Escape] Failed to parse valid actions from VLM response")
        return False
```

#### Change 1.12 — Modify stuck detection block
**Location:** Lines 828–899 (the stuck detection and recovery block)

Wrap the existing logic with an `llm_escape` check:

```python
self.loop_time = 0
# Stuck detection and recovery
print(f"[Act] Stuck detection - not_move_steps: {self.not_move_steps}, "
      f"found_goal: {self.found_goal}, action: {number_action}")

if (not self.found_goal and number_action == 0) or self.not_move_steps >= 7:
    print(f"[Act] Agent stuck! Attempting recovery...")
    
    # Reset goal flags if stationary too long
    if self.not_move_steps >= 7:
        print(f"[Act] Stationary for {self.not_move_steps} steps, resetting goal flags")
        self.found_goal = False
        self.found_possible_goal = False
        self.not_move_steps = 0
    
    # === LLM ESCAPE MODE ===
    if hasattr(self.args, 'llm_escape') and self.args.llm_escape \
            and self.escape_attempts < self.max_escape_attempts:
        if self.llm_plan_escape(observations):
            # First action from escape queue
            number_action = self.escape_action_queue.pop(0)
            print(f"[LLM Escape] Starting escape with action: {number_action}")
        else:
            # VLM failed to produce actions, fall through to primitive recovery
            print(f"[LLM Escape] VLM failed, falling back to primitive recovery")
            <existing primitive recovery code here>
    else:
        # Primitive recovery (existing code, unchanged)
        <existing primitive recovery code lines 843-899>
```

The existing primitive code (blocking stuck region, FBE, random goal, forced turns) stays **exactly as-is** inside the `else` branch.

---

### File 2: `scenegraph.py`

**No changes required.** The existing methods are sufficient:
- `get_llm_response(prompt)` — text-only LLM call (line 814)
- `get_vlm_response(prompt, image)` — VLM call with image (line 824)
- `GroupNode.graph_to_text()` — scene graph text serialization (line 67)
- `get_edges()` — retrieve all edges (line 295)

---

### File 3: `utils/utils_glip.py`

**No changes required.** The `categories_extended` list is already defined and available.

---

### File 4: `run_exploration_mode.py` (if used as alternative entry point)

#### Change 4.1 — Add `--llm_escape` flag to argparse
**Location:** After the existing `add_argument` calls (around line 54)

```python
parser.add_argument(
    "--llm_escape", action='store_true',
    help="Enable LLM-guided escape when robot is stuck"
)
```

---

## Summary of All Changes

| File | What | Lines Affected |
|---|---|---|
| `SG_Nav.py` | Add `--llm_escape` CLI flag | ~1477 (argparse block) |
| `SG_Nav.py` | Add escape state vars in `__init__` | ~165 |
| `SG_Nav.py` | Reset escape state in `reset()` | ~277 |
| `SG_Nav.py` | Reset escape state in `reset_local_scenegraph()` | ~340 |
| `SG_Nav.py` | Track action history | ~911 |
| `SG_Nav.py` | Escape queue execution at top of `act()` | ~564 |
| `SG_Nav.py` | New method: `_serialize_scene_graph_for_vlm()` | new (~535) |
| `SG_Nav.py` | New method: `_get_spatial_map_summary()` | new |
| `SG_Nav.py` | New method: `_build_escape_prompt()` | new |
| `SG_Nav.py` | New method: `_parse_escape_actions()` | new |
| `SG_Nav.py` | New method: `llm_plan_escape()` | new |
| `SG_Nav.py` | Modify stuck detection block (wrap with llm_escape check) | 828–899 |
| `run_exploration_mode.py` | Add `--llm_escape` flag | ~54 |
| `scenegraph.py` | No changes | — |
| `utils/utils_glip.py` | No changes | — |

## Usage

```bash
# Without LLM escape (existing behavior)
nohup python SG_Nav.py > output_run.log 2>&1 &

# With LLM escape enabled
nohup python SG_Nav.py --llm_escape > output_run.log 2>&1 &
```

## Notes

- The VLM call adds latency (~1-3s per escape plan) but only triggers when stuck, so impact on normal navigation is zero
- Escape attempts reset on `reset()` and `reset_local_scenegraph()` (per episode / per target object)
- The 5×5 collision grid in the prompt gives the VLM local spatial awareness without sending the full 800×800 map
- If the VLM produces an unparseable response, it falls through to primitive recovery immediately
- Action history buffer (last 20 actions) helps the VLM see oscillation patterns (e.g., repeated left-right-left-right)
