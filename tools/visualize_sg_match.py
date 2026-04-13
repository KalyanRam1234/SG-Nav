#!/usr/bin/env python3
"""
Scene Graph Match Visualizer — Interactive HTML visualization of sg_match results.

Shows two scene graphs side-by-side with match results overlaid:
  - SAME matches (green), MOVED (orange), UNCERTAIN (yellow)
  - Click any node for full feature details
  - Matched edges shown with snapshot comparison panels
  - Unmatched nodes styled distinctly (ADDED/REMOVED/UNSEEN)

Usage:
    # Run sg_match and visualize in one step
    python tools/visualize_sg_match.py \\
        --sg1 data/usefulruns/run1.pkl \\
        --sg2 data/usefulruns/run2.pkl \\
        --mode dynamic --refine-iters 2

    # Visualize from a saved match report JSON
    python tools/visualize_sg_match.py --report match_report.json

    # Custom output path
    python tools/visualize_sg_match.py --sg1 ... --sg2 ... -o my_match_viz.html
"""

import argparse
import base64
import json
import pickle
import sys
from collections import defaultdict
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from pyvis.network import Network
except ImportError:
    print("pyvis is required: pip install pyvis")
    sys.exit(1)


# ── Color Palette ───────────────────────────────────────────────────────
MATCH_COLORS = {
    'SAME':       '#2ECC71',  # green
    'MOVED':      '#E67E22',  # orange
    'UNCERTAIN':  '#F1C40F',  # yellow
    'REPLACED':   '#E74C3C',  # red
}
UNMATCHED_COLORS = {
    'ADDED':   '#3498DB',     # blue  (in SG2 only)
    'REMOVED': '#95A5A6',     # gray  (in SG1 only)
    'UNSEEN':  '#8E44AD',     # purple
    'REJECTED': '#7F8C8D',    # dark gray
}
ROOM_COLORS = {
    'bedroom':      '#6A5ACD',
    'living room':  '#2E8B57',
    'bathroom':     '#4682B4',
    'kitchen':      '#DAA520',
    'dining room':  '#CD853F',
    'office room':  '#708090',
    'gym':          '#DC143C',
    'lounge':       '#9370DB',
    'laundry room': '#5F9EA0',
}
EDGE_MATCH_COLOR   = '#2ECC71'   # green for consistent edges
EDGE_NOMATCH_COLOR = '#E74C3C'   # red for inconsistent
EDGE_DEFAULT       = '#555555'
BG_COLOR = '#1a1a2e'


# ── Data Loading ────────────────────────────────────────────────────────

def load_scene_graph_data(pkl_path: str) -> dict:
    """Load a scene graph pickle and return the serialized dict."""
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    if isinstance(raw, dict) and 'scenegraph' in raw:
        return raw['scenegraph']
    return raw


def _fmt(val, decimals=3):
    """Format a value for display."""
    if val is None:
        return '—'
    if isinstance(val, float):
        return f'{val:.{decimals}f}'
    if isinstance(val, (list, np.ndarray)):
        arr = np.asarray(val)
        if arr.ndim == 0:
            return f'{float(arr):.{decimals}f}'
        if arr.size <= 6:
            return '[' + ', '.join(f'{x:.2f}' for x in arr.flat) + ']'
        return f'[{arr.shape} array]'
    return str(val)


# ── Node Tooltip Builder ────────────────────────────────────────────────

def _build_node_tooltip(node: dict, sg_label: str, match_info: Optional[dict] = None) -> str:
    """Build rich HTML tooltip for a node."""
    cap = node.get('caption', '?')
    obj = node.get('object', {}) or {}

    lines = [f'<b>{cap}</b> ({sg_label} idx {node.get("_idx", "?")})']

    # Position
    center = node.get('center')
    if center:
        lines.append(f'Position: ({center[0]:.1f}, {center[1]:.1f})')
    c3d = obj.get('centroid')
    if c3d:
        c3d = np.asarray(c3d)
        lines.append(f'3D: ({c3d[0]:.2f}, {c3d[1]:.2f}, {c3d[2]:.2f})')

    # Room
    room = node.get('_room_caption')
    if room:
        lines.append(f'Room: {room}')

    # Detection stats
    n_det = obj.get('n_detections', len(obj.get('conf', [])))
    n_pts = sum(obj.get('n_points', [0]))
    lines.append(f'Detections: {n_det}, Points: {n_pts:,}')

    # Observation metadata (Phase 1)
    first = obj.get('first_seen_step')
    last = obj.get('last_seen_step')
    if first is not None:
        lines.append(f'Seen: step {first} → {last}')
    hr = obj.get('height_range')
    if hr:
        lines.append(f'Height range: {hr[0]:.2f} → {hr[1]:.2f} m')
    oc = obj.get('observation_confidence')
    if oc is not None:
        lines.append(f'Obs confidence: {oc:.3f}')

    # Geometry (Phase 2)
    ext = obj.get('bbox_extent')
    if ext is not None:
        ext = np.asarray(ext)
        lines.append(f'BBox extent: {ext[0]:.2f} × {ext[1]:.2f} × {ext[2]:.2f}')
    fpfh = obj.get('fpfh_descriptor')
    if fpfh is not None:
        lines.append(f'FPFH: {len(fpfh)}-D descriptor ✓')

    # Visuals (Phase 3)
    dc = obj.get('dominant_colors')
    if dc is not None:
        n_colors = len(dc)
        lines.append(f'Dominant colors: {n_colors} clusters')

    # Score
    score = node.get('score', 0)
    if score:
        lines.append(f'Score: {score:.2f}')

    # Match info
    if match_info:
        lines.append('')
        lines.append(f'<b>Match: {match_info["change_type"]}</b>')
        lines.append(f'Cost: {match_info["cost"]:.3f}')
        lines.append(f'CLIP sim: {match_info["clip_visual_sim"]:.3f}')
        lines.append(f'Spatial dist: {match_info["spatial_dist_3d"]:.2f} m')
        lines.append(f'Color sim: {match_info["color_hist_sim"]:.3f}')
        lines.append(f'Edge consist: {match_info["edge_consistency"]:.3f}')
        lines.append(f'Confidence: {match_info["confidence"]:.3f}')
        partner = match_info.get('partner_caption', '?')
        lines.append(f'Partner: {partner}')

    return '<br>'.join(lines)


def _build_unmatched_tooltip(node: dict, sg_label: str, reason: str) -> str:
    """Tooltip for unmatched nodes."""
    cap = node.get('caption', '?')
    lines = [f'<b>{cap}</b> ({sg_label} idx {node.get("_idx", "?")})']
    lines.append(f'<b style="color:{UNMATCHED_COLORS.get(reason, "#999")}">{reason}</b>')
    room = node.get('_room_caption')
    if room:
        lines.append(f'Room: {room}')
    center = node.get('center')
    if center:
        lines.append(f'Position: ({center[0]:.1f}, {center[1]:.1f})')
    return '<br>'.join(lines)


# ── Edge Snapshot Comparison Builder ────────────────────────────────────

def _find_matching_edges(sg1_edges, sg2_edges, match_map_1to2):
    """Find edge pairs between the two graphs via the node match mapping.

    Returns list of (sg1_edge, sg2_edge, is_consistent) tuples.
    """
    # Build SG2 edge lookup: (n1_idx, n2_idx) -> edge
    sg2_edge_lookup = {}
    for e in sg2_edges:
        key = (e['node1_idx'], e['node2_idx'])
        sg2_edge_lookup[key] = e
        sg2_edge_lookup[(e['node2_idx'], e['node1_idx'])] = e

    paired = []
    for e1 in sg1_edges:
        n1a, n1b = e1['node1_idx'], e1['node2_idx']
        # Map to SG2 indices
        n2a = match_map_1to2.get(n1a)
        n2b = match_map_1to2.get(n1b)
        if n2a is None or n2b is None:
            continue
        e2 = sg2_edge_lookup.get((n2a, n2b))
        is_consistent = e2 is not None
        paired.append((e1, e2, is_consistent))
    return paired


# ── Main Build ──────────────────────────────────────────────────────────

def build_match_network(sg1_data, sg2_data, report,
                        title="Scene Graph Match Visualization",
                        show_edges=False):
    """Build a pyvis Network showing both graphs and their match results.

    Args:
        show_edges: If True, draw internal SG1/SG2 edges and enable
                    snapshot comparison. Default False (nodes + match links only).
    """
    nodes1 = sg1_data.get('nodes', [])
    nodes2 = sg2_data.get('nodes', [])
    edges1 = sg1_data.get('edges', [])
    edges2 = sg2_data.get('edges', [])
    rooms1 = sg1_data.get('room_nodes', [])
    rooms2 = sg2_data.get('room_nodes', [])

    # Tag nodes with their index and room caption for tooltip use
    for i, n in enumerate(nodes1):
        n['_idx'] = i
        ri = n.get('room_idx')
        n['_room_caption'] = rooms1[ri]['caption'] if ri is not None and ri < len(rooms1) else None
    for i, n in enumerate(nodes2):
        n['_idx'] = i
        ri = n.get('room_idx')
        n['_room_caption'] = rooms2[ri]['caption'] if ri is not None and ri < len(rooms2) else None

    matches = report.get('matches', [])
    unmatched = report.get('unmatched', [])

    # Build lookup structures
    # sg1_idx -> match info, sg2_idx -> match info
    sg1_matched = {}
    sg2_matched = {}
    match_map_1to2 = {}  # sg1_idx -> sg2_idx
    for m in matches:
        sg1_matched[m['sg1_idx']] = m
        sg2_matched[m['sg2_idx']] = m
        match_map_1to2[m['sg1_idx']] = m['sg2_idx']

    # Unmatched lookup
    sg1_unmatched = {}
    sg2_unmatched = {}
    for u in unmatched:
        if u.get('sg_source') == 'sg1':
            sg1_unmatched[u['node_idx']] = u
        else:
            sg2_unmatched[u['node_idx']] = u

    # ID offsets to avoid collision: SG1 nodes = 0..N, SG2 = 100000+
    SG2_OFFSET = 100000

    net = Network(
        height="950px",
        width="100%",
        bgcolor=BG_COLOR,
        font_color="white",
        directed=False,
        heading=title,
        cdn_resources="remote",
    )
    net.barnes_hut(
        gravity=-5000,
        central_gravity=0.08,
        spring_length=250,
        spring_strength=0.01,
    )

    # ── SG1 Nodes (left side) ──
    for i, node in enumerate(nodes1):
        cap = node.get('caption', f'n_{i}')
        m = sg1_matched.get(i)
        u = sg1_unmatched.get(i)

        if m:
            ct = m['change_type']
            color = MATCH_COLORS.get(ct, '#AAAAAA')
            m_info = dict(m)
            m_info['partner_caption'] = m.get('sg2_caption', '?')
            tooltip = _build_node_tooltip(node, 'SG1', m_info)
            border_w = 3
        elif u:
            ct = u.get('change_type', 'REMOVED')
            color = UNMATCHED_COLORS.get(ct, '#7F8C8D')
            tooltip = _build_unmatched_tooltip(node, 'SG1', ct)
            border_w = 1
        else:
            color = '#7F8C8D'
            tooltip = _build_unmatched_tooltip(node, 'SG1', 'SIZE_ASYM')
            border_w = 1

        net.add_node(
            i,
            label=f'①{cap}',
            title=tooltip,
            color={'background': color, 'border': color,
                   'highlight': {'background': '#FFD700', 'border': '#FF4500'}},
            shape='dot',
            size=18 if m else 12,
            borderWidth=border_w,
            font={'size': 10, 'color': 'white'},
            x=-600 + np.random.randint(-100, 100),
            y=i * 5,
        )

    # ── SG2 Nodes (right side) ──
    for i, node in enumerate(nodes2):
        cap = node.get('caption', f'n_{i}')
        m = sg2_matched.get(i)
        u = sg2_unmatched.get(i)

        if m:
            ct = m['change_type']
            color = MATCH_COLORS.get(ct, '#AAAAAA')
            m_info = dict(m)
            m_info['partner_caption'] = m.get('sg1_caption', '?')
            tooltip = _build_node_tooltip(node, 'SG2', m_info)
            border_w = 3
        elif u:
            ct = u.get('change_type', 'ADDED')
            color = UNMATCHED_COLORS.get(ct, '#3498DB')
            tooltip = _build_unmatched_tooltip(node, 'SG2', ct)
            border_w = 1
        else:
            color = '#3498DB'
            tooltip = _build_unmatched_tooltip(node, 'SG2', 'SIZE_ASYM')
            border_w = 1

        nid = SG2_OFFSET + i
        net.add_node(
            nid,
            label=f'②{cap}',
            title=tooltip,
            color={'background': color, 'border': color,
                   'highlight': {'background': '#FFD700', 'border': '#FF4500'}},
            shape='dot',
            size=18 if m else 12,
            borderWidth=border_w,
            font={'size': 10, 'color': 'white'},
            x=600 + np.random.randint(-100, 100),
            y=i * 5,
        )

    # ── SG1 internal edges (optional) ──
    if show_edges:
        for e in edges1:
            n1, n2 = e['node1_idx'], e['node2_idx']
            rel = e.get('relation', '')
            net.add_edge(n1, n2,
                         title=f'SG1: {rel}',
                         color='#444466', width=1,
                         font={'size': 8, 'color': '#777'})

        # ── SG2 internal edges (optional) ──
        for e in edges2:
            n1, n2 = e['node1_idx'], e['node2_idx']
            rel = e.get('relation', '')
            net.add_edge(SG2_OFFSET + n1, SG2_OFFSET + n2,
                         title=f'SG2: {rel}',
                         color='#444466', width=1,
                         font={'size': 8, 'color': '#777'})

    # ── Cross-graph match edges ──
    for m in matches:
        ct = m['change_type']
        color = MATCH_COLORS.get(ct, '#AAAAAA')
        dist = m.get('spatial_dist_3d', 0)
        clip = m.get('clip_visual_sim', 0)
        conf = m.get('confidence', 0)

        label = ct
        title_parts = [
            f'{m["sg1_caption"]} ↔ {m["sg2_caption"]}',
            f'Type: {ct}',
            f'Cost: {m.get("cost", 0):.3f}',
            f'CLIP: {clip:.3f}',
            f'Spatial: {dist:.2f}m',
            f'Color: {m.get("color_hist_sim", 0):.3f}',
            f'Edge consistency: {m.get("edge_consistency", 0):.3f}',
            f'Confidence: {conf:.3f}',
        ]
        if not m.get('same_room', True):
            title_parts.append(f'⚠ Room mismatch: {m.get("sg1_room")} vs {m.get("sg2_room")}')

        width = 4 if ct == 'SAME' else 3
        dashes = ct in ('MOVED', 'UNCERTAIN')

        net.add_edge(
            m['sg1_idx'],
            SG2_OFFSET + m['sg2_idx'],
            label=label,
            title='<br>'.join(title_parts),
            color=color,
            width=width,
            dashes=dashes,
            font={'size': 10, 'color': color, 'strokeWidth': 0},
        )

    # ── Collect edge snapshot pairs for the comparison panel (optional) ──
    snapshot_pairs = {}
    if show_edges:
        edge_pairs = _find_matching_edges(edges1, edges2, match_map_1to2)
        for e1, e2, is_consistent in edge_pairs:
            n1a, n1b = e1['node1_idx'], e1['node2_idx']
            cap1a = nodes1[n1a].get('caption', '?') if n1a < len(nodes1) else '?'
            cap1b = nodes1[n1b].get('caption', '?') if n1b < len(nodes1) else '?'
            rel1 = e1.get('relation', '?')

            entry = {
                'sg1_b64': e1.get('snapshot_b64'),
                'sg1_relation': rel1,
                'sg1_cap': f'{cap1a} —{rel1}— {cap1b}',
                'sg1_step': e1.get('snapshot_step', '?'),
                'sg2_b64': None,
                'sg2_relation': None,
                'sg2_cap': 'No matching edge',
                'sg2_step': None,
                'consistent': is_consistent,
            }
            if e2:
                n2a, n2b = e2['node1_idx'], e2['node2_idx']
                cap2a = nodes2[n2a].get('caption', '?') if n2a < len(nodes2) else '?'
                cap2b = nodes2[n2b].get('caption', '?') if n2b < len(nodes2) else '?'
                rel2 = e2.get('relation', '?')
                entry['sg2_b64'] = e2.get('snapshot_b64')
                entry['sg2_relation'] = rel2
                entry['sg2_cap'] = f'{cap2a} —{rel2}— {cap2b}'
                entry['sg2_step'] = e2.get('snapshot_step', '?')

            if entry['sg1_b64'] or entry['sg2_b64']:
                key = f'{n1a}-{n1b}'
                snapshot_pairs[key] = entry

    net._snapshot_pairs = snapshot_pairs
    net._report_summary = report.get('summary', {})
    net._accuracy = report.get('accuracy', {})
    net._edge_detail = report.get('edge_consistency_detail', {})

    return net


# ── HTML Injection ──────────────────────────────────────────────────────

def _inject_match_ui(html_path: str, net):
    """Inject CSS, legend, stats panel, snapshot comparison modal, and
    detail sidebar into the pyvis HTML."""
    snapshot_pairs = getattr(net, '_snapshot_pairs', {})
    summary = getattr(net, '_report_summary', {})
    accuracy = getattr(net, '_accuracy', {})
    edge_detail = getattr(net, '_edge_detail', {})

    # Build safe snapshot JSON (strip large b64 for the JS object)
    snapshot_json = json.dumps({
        k: {
            'sg1_b64': v['sg1_b64'],
            'sg1_cap': v['sg1_cap'],
            'sg1_step': v['sg1_step'],
            'sg2_b64': v['sg2_b64'],
            'sg2_cap': v['sg2_cap'],
            'sg2_step': v['sg2_step'],
            'consistent': v['consistent'],
        }
        for k, v in snapshot_pairs.items()
    })

    # Stats summary
    n_same = summary.get('SAME', 0)
    n_moved = summary.get('MOVED', 0)
    n_uncertain = summary.get('UNCERTAIN', 0)
    n_total = n_same + n_moved + n_uncertain
    mas = accuracy.get('score', 0)
    ec_soft = edge_detail.get('soft', 0)
    ec_hard = edge_detail.get('hard', 0)

    injection = f"""
<!-- Match Visualization UI -->
<style>
  #matchLegend {{
    position: fixed; top: 10px; left: 10px; z-index: 1000;
    background: rgba(26,26,46,0.92); border: 1px solid #444;
    border-radius: 8px; padding: 12px 16px;
    font-family: 'Segoe UI', monospace; font-size: 12px; color: #eee;
    max-width: 280px;
  }}
  #matchLegend h3 {{ margin: 0 0 8px; font-size: 14px; color: #fff; }}
  .legend-row {{ display: flex; align-items: center; margin: 3px 0; }}
  .legend-dot {{
    width: 14px; height: 14px; border-radius: 50%; margin-right: 8px;
    flex-shrink: 0; border: 1px solid rgba(255,255,255,0.3);
  }}
  .legend-line {{
    width: 24px; height: 3px; margin-right: 8px; flex-shrink: 0;
  }}
  .legend-dashed {{
    background: repeating-linear-gradient(90deg, currentColor 0, currentColor 6px, transparent 6px, transparent 12px);
    height: 3px; width: 24px; margin-right: 8px; flex-shrink: 0;
  }}
  #statsPanel {{
    position: fixed; top: 10px; right: 10px; z-index: 1000;
    background: rgba(26,26,46,0.92); border: 1px solid #444;
    border-radius: 8px; padding: 12px 16px;
    font-family: monospace; font-size: 12px; color: #eee;
    min-width: 220px;
  }}
  #statsPanel h3 {{ margin: 0 0 8px; font-size: 14px; color: #fff; }}
  .stat-row {{ display: flex; justify-content: space-between; margin: 2px 0; }}
  .stat-val {{ color: #2ECC71; font-weight: bold; }}
  .stat-warn {{ color: #E67E22; }}

  #snapshotModal {{
    display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.9); z-index: 9999;
    justify-content: center; align-items: center; flex-direction: column;
    cursor: pointer;
  }}
  #snapshotModal.active {{ display: flex; }}
  .snap-compare {{
    display: flex; gap: 20px; align-items: flex-start;
    max-width: 90%; max-height: 70%;
  }}
  .snap-panel {{
    text-align: center; flex: 1;
  }}
  .snap-panel img {{
    max-width: 100%; max-height: 55vh; border-radius: 6px;
    border: 2px solid #555;
  }}
  .snap-panel .snap-label {{
    color: #fff; font-family: monospace; font-size: 13px;
    margin-top: 8px; padding: 4px 8px; border-radius: 4px;
    display: inline-block;
  }}
  .snap-consistent {{ border-color: #2ECC71 !important; }}
  .snap-inconsistent {{ border-color: #E74C3C !important; }}
  .snap-status {{
    color: #fff; font-family: monospace; font-size: 16px;
    margin-top: 12px; padding: 6px 16px; border-radius: 4px;
  }}
  .snap-status.consistent {{ background: rgba(46,204,113,0.3); }}
  .snap-status.inconsistent {{ background: rgba(231,76,60,0.3); }}
  #snapHint {{ color: #888; font-size: 11px; margin-top: 8px; }}

  #detailSidebar {{
    display: none; position: fixed; top: 50px; right: 10px; z-index: 1001;
    background: rgba(26,26,46,0.95); border: 1px solid #2ECC71;
    border-radius: 8px; padding: 14px 18px;
    font-family: monospace; font-size: 11px; color: #eee;
    max-width: 340px; max-height: 80vh; overflow-y: auto;
  }}
  #detailSidebar.active {{ display: block; }}
  #detailSidebar h3 {{ margin: 0 0 6px; color: #2ECC71; font-size: 13px; }}
  #detailSidebar .close-btn {{
    position: absolute; top: 6px; right: 10px; cursor: pointer;
    color: #999; font-size: 16px;
  }}
  #detailSidebar .close-btn:hover {{ color: #fff; }}
  .detail-section {{ margin-top: 8px; border-top: 1px solid #333; padding-top: 6px; }}
  .detail-section h4 {{ margin: 0 0 4px; color: #F1C40F; font-size: 12px; }}
  .detail-kv {{ display: flex; justify-content: space-between; padding: 1px 0; }}
  .detail-kv .dk {{ color: #999; }}
  .detail-kv .dv {{ color: #eee; text-align: right; max-width: 180px; overflow: hidden;
                    text-overflow: ellipsis; white-space: nowrap; }}
</style>

<!-- Legend -->
<div id="matchLegend">
  <h3>Match Legend</h3>
  <div class="legend-row"><div class="legend-dot" style="background:#2ECC71"></div> SAME (matched, same position)</div>
  <div class="legend-row"><div class="legend-dot" style="background:#E67E22"></div> MOVED (matched, relocated)</div>
  <div class="legend-row"><div class="legend-dot" style="background:#F1C40F"></div> UNCERTAIN (low confidence)</div>
  <div class="legend-row"><div class="legend-dot" style="background:#3498DB"></div> ADDED / Size asymmetry (SG2)</div>
  <div class="legend-row"><div class="legend-dot" style="background:#95A5A6"></div> REMOVED / Size asymmetry (SG1)</div>
  <div class="legend-row"><div class="legend-dot" style="background:#8E44AD"></div> UNSEEN (unexplored area)</div>
  <hr style="border-color:#444; margin:6px 0">
  <div class="legend-row"><div class="legend-line" style="background:#2ECC71"></div> SAME match link</div>
  <div class="legend-row"><div class="legend-dashed" style="color:#E67E22"></div> MOVED match link</div>
  <div class="legend-row"><div class="legend-dashed" style="color:#F1C40F"></div> UNCERTAIN match link</div>
  <div class="legend-row"><div class="legend-line" style="background:#444466"></div> Internal edge (within SG)</div>
  <hr style="border-color:#444; margin:6px 0">
  <div style="color:#888; font-size:10px">① = SG1 node &nbsp; ② = SG2 node<br>
  Click node for details · Click green SG1 edge for snapshot comparison</div>
</div>

<!-- Stats Panel -->
<div id="statsPanel">
  <h3>Match Statistics</h3>
  <div class="stat-row"><span>MAS</span><span class="stat-val">{mas:.1%}</span></div>
  <div class="stat-row"><span>SAME</span><span class="stat-val">{n_same}</span></div>
  <div class="stat-row"><span>MOVED</span><span class="stat-warn">{n_moved}</span></div>
  <div class="stat-row"><span>UNCERTAIN</span><span style="color:#F1C40F">{n_uncertain}</span></div>
  <div class="stat-row"><span>Total matched</span><span>{n_total}</span></div>
  <hr style="border-color:#444; margin:6px 0">
  <div class="stat-row"><span>Edge consist (soft)</span><span class="stat-val">{ec_soft:.1%}</span></div>
  <div class="stat-row"><span>Edge consist (hard)</span><span>{ec_hard:.1%}</span></div>
  <div class="stat-row"><span>Snapshot pairs</span><span>{len(snapshot_pairs)}</span></div>
</div>

<!-- Snapshot Comparison Modal -->
<div id="snapshotModal" onclick="this.classList.remove('active')">
  <div class="snap-compare" onclick="event.stopPropagation()">
    <div class="snap-panel">
      <div style="color:#87CEEB; margin-bottom:4px; font-family:monospace">SG1</div>
      <img id="snap1Img" class="snap-consistent" />
      <div class="snap-label" id="snap1Label"></div>
    </div>
    <div class="snap-panel">
      <div style="color:#87CEEB; margin-bottom:4px; font-family:monospace">SG2</div>
      <img id="snap2Img" class="snap-consistent" />
      <div class="snap-label" id="snap2Label"></div>
    </div>
  </div>
  <div class="snap-status" id="snapStatus"></div>
  <div id="snapHint">Click outside to close</div>
</div>

<!-- Detail Sidebar (populated on node click) -->
<div id="detailSidebar">
  <span class="close-btn" onclick="document.getElementById('detailSidebar').classList.remove('active')">✕</span>
  <div id="detailContent"></div>
</div>

<script>
var snapshotPairs = {snapshot_json};

// Wait for vis.js network
var _waitNet2 = setInterval(function() {{
    if (typeof network === 'undefined') return;
    clearInterval(_waitNet2);

    // Edge click -> snapshot comparison
    network.on("selectEdge", function(params) {{
        if (params.edges.length === 0) return;
        var edgeId = params.edges[0];
        var ed = network.body.data.edges.get(edgeId);
        if (!ed) return;
        // Only match SG1 internal edges (both nodes < 100000)
        var from = ed.from, to = ed.to;
        if (from >= 100000 || to >= 100000) return;
        var key = from + "-" + to;
        var snap = snapshotPairs[key];
        if (!snap) {{
            key = to + "-" + from;
            snap = snapshotPairs[key];
        }}
        if (!snap) return;

        var img1 = document.getElementById('snap1Img');
        var img2 = document.getElementById('snap2Img');
        var lab1 = document.getElementById('snap1Label');
        var lab2 = document.getElementById('snap2Label');
        var status = document.getElementById('snapStatus');

        if (snap.sg1_b64) {{
            img1.src = 'data:image/jpeg;base64,' + snap.sg1_b64;
            img1.style.display = '';
        }} else {{
            img1.style.display = 'none';
        }}
        lab1.textContent = snap.sg1_cap + (snap.sg1_step ? ' (step ' + snap.sg1_step + ')' : '');

        if (snap.sg2_b64) {{
            img2.src = 'data:image/jpeg;base64,' + snap.sg2_b64;
            img2.style.display = '';
            img2.className = snap.consistent ? 'snap-consistent' : 'snap-inconsistent';
        }} else {{
            img2.style.display = 'none';
        }}
        lab2.textContent = snap.sg2_cap + (snap.sg2_step ? ' (step ' + snap.sg2_step + ')' : '');

        if (snap.consistent) {{
            status.textContent = '✓ Edge preserved in both graphs';
            status.className = 'snap-status consistent';
        }} else {{
            status.textContent = '✗ Edge NOT found in SG2';
            status.className = 'snap-status inconsistent';
        }}
        document.getElementById('snapshotModal').classList.add('active');
    }});

    // Node click -> detail sidebar
    network.on("selectNode", function(params) {{
        if (params.nodes.length === 0) return;
        var nid = params.nodes[0];
        var nodeData = network.body.data.nodes.get(nid);
        if (!nodeData || !nodeData.title) return;
        document.getElementById('detailContent').innerHTML =
            '<h3>' + nodeData.label + '</h3>' + nodeData.title;
        document.getElementById('detailSidebar').classList.add('active');
    }});

    network.on("deselectNode", function() {{
        document.getElementById('detailSidebar').classList.remove('active');
    }});
}}, 200);
</script>
"""

    with open(html_path, 'r') as f:
        html = f.read()
    html = html.replace('</body>', injection + '\n</body>')
    with open(html_path, 'w') as f:
        f.write(html)


# ── Public API ──────────────────────────────────────────────────────────

def visualize_match(sg1_path, sg2_path, report_dict,
                    output_path='sg_match_viz.html',
                    title='SG-Nav Match Visualization',
                    show_edges=False):
    """Generate interactive HTML visualization of match results.

    Args:
        sg1_path: path to SG1 pickle
        sg2_path: path to SG2 pickle
        report_dict: MatchReport as dict (from asdict() or JSON)
        output_path: where to write HTML
        title: page title
        show_edges: if True, draw internal edges and enable snapshot comparison
    """
    sg1 = load_scene_graph_data(sg1_path)
    sg2 = load_scene_graph_data(sg2_path)

    net = build_match_network(sg1, sg2, report_dict, title=title,
                              show_edges=show_edges)

    output_path = str(output_path)
    net.show(output_path, notebook=False)

    _inject_match_ui(output_path, net)

    n_matches = len(report_dict.get('matches', []))
    n_snaps = len(getattr(net, '_snapshot_pairs', {}))
    mas = report_dict.get('accuracy', {}).get('score', 0)
    print(f"Match visualization saved to: {output_path}")
    print(f"  {n_matches} match links, {n_snaps} snapshot pairs, MAS={mas:.1%}")
    return output_path


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Visualize sg_match results as interactive HTML.')
    parser.add_argument('--sg1', type=str, help='Path to SG1 pickle')
    parser.add_argument('--sg2', type=str, help='Path to SG2 pickle')
    parser.add_argument('--report', type=str, default=None,
                        help='Pre-computed match report JSON (skip matching)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output HTML path')
    parser.add_argument('--mode', default='static',
                        choices=['static', 'dynamic'],
                        help='Matching mode preset')
    parser.add_argument('--refine-iters', type=int, default=0)
    parser.add_argument('--show-edges', action='store_true', default=False,
                        help='Show internal SG edges and snapshot comparison (default: off)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.report:
        # Load pre-computed report
        with open(args.report) as f:
            report_dict = json.load(f)
        sg1_path = report_dict.get('sg1_path', args.sg1)
        sg2_path = report_dict.get('sg2_path', args.sg2)
        if not sg1_path or not sg2_path:
            print("Error: --sg1 and --sg2 required when report doesn't contain paths")
            sys.exit(1)
    else:
        if not args.sg1 or not args.sg2:
            print("Error: --sg1 and --sg2 are required (or provide --report)")
            sys.exit(1)
        sg1_path = args.sg1
        sg2_path = args.sg2

        # Import and run matching
        from tools.sg_match import MatchWeights, ChangeThresholds, match_scene_graphs
        from dataclasses import asdict

        weights = MatchWeights.preset(args.mode)
        report = match_scene_graphs(
            sg1_path, sg2_path,
            weights=weights,
            thresholds=ChangeThresholds(),
            verbose=args.verbose,
            refine_iters=args.refine_iters,
        )
        report_dict = asdict(report)

    # Determine output path
    if args.output:
        out = args.output
    else:
        stem1 = Path(sg1_path).stem[:30]
        stem2 = Path(sg2_path).stem[:30]
        out = f'sg_match_{stem1}_vs_{stem2}.html'

    title = f'SG Match: {Path(sg1_path).stem[:40]} vs {Path(sg2_path).stem[:40]}'
    visualize_match(sg1_path, sg2_path, report_dict,
                    output_path=out, title=title,
                    show_edges=args.show_edges)


if __name__ == '__main__':
    main()
