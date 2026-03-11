#!/usr/bin/env python3
"""
Scene Graph Visualizer — Interactive HTML visualization using pyvis.

Usage:
    # Visualize a saved scene graph pickle file
    python tools/visualize_scenegraph.py path/to/global_sg_*.pkl

    # Visualize with custom output path
    python tools/visualize_scenegraph.py path/to/global_sg_*.pkl -o my_graph.html

    # Visualize the live scene graph object directly (from code)
    from tools.visualize_scenegraph import visualize_scenegraph
    visualize_scenegraph(scenegraph, output_path="scene_graph.html")
"""

import argparse
import pickle
import sys
from pathlib import Path

try:
    from pyvis.network import Network
except ImportError:
    print("pyvis is required: pip install pyvis")
    sys.exit(1)


# Color palette for room types
ROOM_COLORS = {
    'bedroom':      '#6A5ACD',  # slate blue
    'living room':  '#2E8B57',  # sea green
    'bathroom':     '#4682B4',  # steel blue
    'kitchen':      '#DAA520',  # goldenrod
    'dining room':  '#CD853F',  # peru
    'office room':  '#708090',  # slate gray
    'gym':          '#DC143C',  # crimson
    'lounge':       '#9370DB',  # medium purple
    'laundry room': '#5F9EA0',  # cadet blue
}
DEFAULT_ROOM_COLOR = '#A9A9A9'  # dark gray for unknown/unassigned

GOAL_NODE_COLOR = '#FF4500'     # orange-red
DEFAULT_NODE_COLOR = '#87CEEB'  # sky blue
ROOM_NODE_COLOR = '#FFD700'     # gold


def build_pyvis_network(data, title="Scene Graph"):
    """Build a pyvis Network from a serialized scene graph dict.

    Args:
        data: dict from SceneGraph.to_serializable_dict() or the full
              save_data dict (with 'scenegraph' key).
        title: title shown in the HTML page.

    Returns:
        pyvis.network.Network instance ready for .show()
    """
    # Accept either the raw sg dict or the full save wrapper
    if 'scenegraph' in data:
        sg = data['scenegraph']
        episode = data.get('episode', '?')
        steps = data.get('total_steps', '?')
        title = f"{title} — Episode {episode}, {steps} steps"
    else:
        sg = data

    nodes = sg.get('nodes', [])
    edges = sg.get('edges', [])
    room_nodes = sg.get('room_nodes', [])
    group_nodes = sg.get('group_nodes', [])

    net = Network(
        height="900px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=False,
        heading=title,
        cdn_resources="remote",
    )
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.15,
        spring_length=400,
        spring_strength=0.02,
    )
    # net.set_options("""
    # {
    #     "layout": {
    #         "improvedLayout": false
    #     },
    #     "physics": {
    #         "barnesHut": {
    #             "gravitationalConstant": -8000,
    #             "centralGravity": 0.15,
    #             "springLength": 400,
    #             "springConstant": 0.02
    #         },
    #         "stabilization": {
    #             "iterations": 200
    #         }
    #     }
    # }
    # """)

    # --- Room nodes (large, gold) ---
    room_id_offset = 10000  # avoid collision with object node IDs
    active_rooms = set()
    for node in nodes:
        room_idx = node.get('room_idx')
        if room_idx is not None:
            active_rooms.add(room_idx)

    for idx in active_rooms:
        if idx >= len(room_nodes):
            continue
        rn = room_nodes[idx]
        caption = rn['caption']
        color = ROOM_COLORS.get(caption, DEFAULT_ROOM_COLOR)
        net.add_node(
            room_id_offset + idx,
            label=caption.upper(),
            title=f"Room: {caption}\nExploration level: {rn.get('exploration_level', 0)}",
            color=color,
            shape="diamond",
            size=35,
            font={"size": 16, "color": "white", "bold": True},
        )

    # --- Group nodes (medium, hexagonal) ---
    group_id_offset = 20000  # avoid collision with room and object IDs
    print(f"Adding {len(group_nodes)} group nodes...")
    for gi, gn in enumerate(group_nodes):
        room_idx = gn.get('room_idx')
        member_idxs = gn.get('node_idxs', [])
        caption = gn.get('caption', f'group_{gi}')
        # Truncate long captions for the label
        label = f"Group {gi}"
        corr_score = gn.get('corr_score', 0)

        if room_idx is not None and room_idx < len(room_nodes):
            room_caption = room_nodes[room_idx]['caption']
            color = ROOM_COLORS.get(room_caption, DEFAULT_ROOM_COLOR)
        else:
            color = DEFAULT_ROOM_COLOR

        tooltip_parts = [f"Group {gi}"]
        tooltip_parts.append(f"Members: {len(member_idxs)}")
        tooltip_parts.append(f"Corr score: {corr_score:.2f}")
        if room_idx is not None and room_idx < len(room_nodes):
            tooltip_parts.append(f"Room: {room_nodes[room_idx]['caption']}")
        tooltip_parts.append(f"Description: {caption}")

        net.add_node(
            group_id_offset + gi,
            label=label,
            title="\n".join(tooltip_parts),
            color=color,
            shape="hexagon",
            size=25,
            font={"size": 13, "color": "white"},
        )

        # Edge from group to its room
        if room_idx is not None and room_idx in active_rooms:
            net.add_edge(
                group_id_offset + gi,
                room_id_offset + room_idx,
                color="#777777",
                width=2,
                dashes=True,
            )

        # Edges from member objects to this group
        for obj_idx in member_idxs:
            if obj_idx < len(nodes):
                net.add_edge(
                    obj_idx,
                    group_id_offset + gi,
                    color="#666666",
                    width=1,
                    dashes=[2, 4],
                )

    # --- Object nodes ---
    grouped_obj_idxs = set()
    for gn in group_nodes:
        grouped_obj_idxs.update(gn.get('node_idxs', []))

    for i, node in enumerate(nodes):
        caption = node.get('caption', f'node_{i}')
        center = node.get('center')
        is_goal = node.get('is_goal_node', False)
        score = node.get('score', 0)
        room_idx = node.get('room_idx')
        exploration = node.get('exploration_level', 0)

        # Pick color
        if is_goal:
            color = GOAL_NODE_COLOR
        elif room_idx is not None and room_idx < len(room_nodes):
            room_caption = room_nodes[room_idx]['caption']
            color = ROOM_COLORS.get(room_caption, DEFAULT_NODE_COLOR)
        else:
            color = DEFAULT_NODE_COLOR

        # Hover tooltip
        tooltip_parts = [f"Object: {caption}"]
        if center:
            tooltip_parts.append(f"Position: ({center[0]:.0f}, {center[1]:.0f})")
        tooltip_parts.append(f"Score: {score:.2f}")
        tooltip_parts.append(f"Exploration: {exploration}")
        if is_goal:
            tooltip_parts.append("⭐ GOAL NODE")
        if room_idx is not None and room_idx < len(room_nodes):
            tooltip_parts.append(f"Room: {room_nodes[room_idx]['caption']}")
        reason = node.get('reason')
        if reason:
            tooltip_parts.append(f"Reason: {reason}")
        tooltip = "\n".join(tooltip_parts)

        border_width = 4 if is_goal else 1
        net.add_node(
            i,
            label=caption,
            title=tooltip,
            color={
                "background": color,
                "border": GOAL_NODE_COLOR if is_goal else color,
                "highlight": {"background": "#FFD700", "border": "#FF4500"},
            },
            shape="dot",
            size=20 + int(score * 15),
            borderWidth=border_width,
            font={"size": 12, "color": "white"},
        )

        # Edge from object to its room (only if not already connected via a group)
        if i not in grouped_obj_idxs and room_idx is not None and room_idx in active_rooms:
            net.add_edge(
                i,
                room_id_offset + room_idx,
                color="#555555",
                width=1,
                dashes=True,
            )

    # --- Spatial relation edges ---
    edge_snapshots = {}  # edge_id -> base64 image data
    for ei, edge in enumerate(edges):
        n1 = edge.get('node1_idx')
        n2 = edge.get('node2_idx')
        relation = edge.get('relation', '')
        if n1 is None or n2 is None:
            continue
        if n1 >= len(nodes) or n2 >= len(nodes):
            continue

        caption1 = nodes[n1].get('caption', '?')
        caption2 = nodes[n2].get('caption', '?')
        has_snap = 'snapshot_b64' in edge

        title_parts = [f"{caption1} —{relation}— {caption2}"]
        if has_snap:
            title_parts.append(f"Step: {edge.get('snapshot_step', '?')}")
            title_parts.append("📷 Click to view snapshot")

        edge_id = f"{n1}-{n2}"
        if has_snap:
            edge_snapshots[edge_id] = {
                'b64': edge['snapshot_b64'],
                'caption1': caption1,
                'caption2': caption2,
                'relation': relation,
                'step': edge.get('snapshot_step', '?'),
            }

        net.add_edge(
            n1,
            n2,
            label=relation if relation else "",
            title="\n".join(title_parts),
            color="#00FF88" if has_snap else "#AAAAAA",
            width=3 if has_snap else 2,
            font={"size": 10, "color": "#CCCCCC", "align": "middle"},
        )

    # Store snapshot data for injection into HTML
    net._edge_snapshots = edge_snapshots

    return net


def visualize_scenegraph(scenegraph_or_path, output_path="scene_graph.html", title="Scene Graph"):
    """Visualize a scene graph and save as interactive HTML.

    Args:
        scenegraph_or_path: Either a SceneGraph object (with to_serializable_dict()),
                            a dict from to_serializable_dict(), or a path to a .pkl file.
        output_path: where to write the HTML file.
        title: page title.

    Returns:
        Path to the generated HTML file.
    """
    if isinstance(scenegraph_or_path, (str, Path)):
        path = Path(scenegraph_or_path)
        print(f"Loading scene graph from {path}...")
        with open(path, 'rb') as f:
            data = pickle.load(f)
    elif hasattr(scenegraph_or_path, 'to_serializable_dict'):
        data = scenegraph_or_path.to_serializable_dict()
    elif isinstance(scenegraph_or_path, dict):
        data = scenegraph_or_path
    else:
        raise TypeError(f"Expected path, dict, or SceneGraph object, got {type(scenegraph_or_path)}")

    net = build_pyvis_network(data, title=title)

    # Count stats for summary
    sg = data.get('scenegraph', data)
    n_nodes = len(sg.get('nodes', []))
    n_edges = len(sg.get('edges', []))
    n_rooms = len([r for i, r in enumerate(sg.get('room_nodes', []))
                   if any(n.get('room_idx') == i for n in sg.get('nodes', []))])

    n_groups = len(sg.get('group_nodes', []))
    n_snapshots = len(getattr(net, '_edge_snapshots', {}))

    output_path = str(output_path)
    net.show(output_path, notebook=False)

    # Inject snapshot modal viewer into the generated HTML
    edge_snapshots = getattr(net, '_edge_snapshots', {})
    if edge_snapshots:
        _inject_snapshot_viewer(output_path, edge_snapshots)

    print(f"Scene graph visualization saved to: {output_path}")
    print(f"  Nodes: {n_nodes}, Edges: {n_edges}, Groups: {n_groups}, Active rooms: {n_rooms}")
    if n_snapshots:
        print(f"  Edge snapshots: {n_snapshots} (green edges — click to view)")
    return output_path


def _inject_snapshot_viewer(html_path, edge_snapshots):
    """Inject CSS/JS into the pyvis HTML to show snapshot images on edge click."""
    import json

    snapshot_json = json.dumps({
        k: {'b64': v['b64'], 'caption1': v['caption1'], 'caption2': v['caption2'],
             'relation': v['relation'], 'step': v['step']}
        for k, v in edge_snapshots.items()
    })

    injection = f"""
<!-- Snapshot Modal Viewer -->
<style>
#snapshotOverlay {{
    display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.85); z-index: 9999; justify-content: center; align-items: center;
    flex-direction: column; cursor: pointer;
}}
#snapshotOverlay.active {{ display: flex; }}
#snapshotOverlay img {{
    max-width: 80%; max-height: 70%; border: 3px solid #00FF88; border-radius: 8px;
    box-shadow: 0 0 40px rgba(0,255,136,0.3);
}}
#snapshotCaption {{
    color: #fff; font-family: monospace; font-size: 16px; margin-top: 16px;
    text-align: center; max-width: 80%;
}}
#snapshotHint {{
    color: #888; font-family: monospace; font-size: 12px; margin-top: 8px;
}}
</style>
<div id="snapshotOverlay" onclick="this.classList.remove('active')">
    <img id="snapshotImg" src="" />
    <div id="snapshotCaption"></div>
    <div id="snapshotHint">Click anywhere to close</div>
</div>
<script>
var edgeSnapshots = {snapshot_json};
// Wait for vis.js network to be ready
var _waitNet = setInterval(function() {{
    if (typeof network !== 'undefined') {{
        clearInterval(_waitNet);
        network.on("selectEdge", function(params) {{
            if (params.edges.length === 0) return;
            var edgeId = params.edges[0];
            var edgeData = network.body.data.edges.get(edgeId);
            if (!edgeData) return;
            var key = edgeData.from + "-" + edgeData.to;
            var snap = edgeSnapshots[key];
            if (!snap) {{
                // Try reverse direction
                key = edgeData.to + "-" + edgeData.from;
                snap = edgeSnapshots[key];
            }}
            if (snap) {{
                document.getElementById('snapshotImg').src = 'data:image/jpeg;base64,' + snap.b64;
                document.getElementById('snapshotCaption').textContent =
                    snap.caption1 + ' \\u2014' + snap.relation + '\\u2014 ' + snap.caption2 +
                    '  (step ' + snap.step + ')';
                document.getElementById('snapshotOverlay').classList.add('active');
            }}
        }});
    }}
}}, 200);
</script>
"""

    with open(html_path, 'r') as f:
        html = f.read()
    html = html.replace('</body>', injection + '\n</body>')
    with open(html_path, 'w') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a saved scene graph as an interactive HTML file."
    )
    parser.add_argument(
        "input",
        help="Path to a scene graph pickle file (global_sg_*.pkl)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output HTML path (default: <input_stem>_viz.html in same directory)",
    )
    parser.add_argument(
        "-t", "--title",
        default="SG-Nav Scene Graph",
        help="Page title",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    if args.output is None:
        output_path = input_path.with_name(input_path.stem + "_viz.html")
    else:
        output_path = Path(args.output)

    visualize_scenegraph(input_path, output_path=output_path, title=args.title)


if __name__ == "__main__":
    main()
