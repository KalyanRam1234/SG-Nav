from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .sg_cache import SerializedSceneGraph, node_aabb_world, node_centroid_world


@dataclass
class SettledPlacement:
    position: List[float]
    rotation_quat_xyzw: Optional[List[float]] = None
    settled: bool = True
    steps: int = 0


def propose_drop_start(
    support_node: Dict[str, Any],
    drop_height: float = 0.4,
    xy_jitter: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> List[float]:
    """Compute a plausible start position above a support for physics drop.

    Uses support AABB to place above the top surface (Y-up).
    """

    c = node_centroid_world(support_node)
    aabb = node_aabb_world(support_node)
    if c is None and aabb is None:
        raise ValueError("Support node has no 3D points (pcd_points/bbox_points)")

    if aabb is not None:
        mn, mx = aabb
        x = float((mn[0] + mx[0]) * 0.5)
        z = float((mn[2] + mx[2]) * 0.5)
        top_y = float(mx[1])
    else:
        x, top_y, z = float(c[0]), float(c[1]), float(c[2])

    if xy_jitter > 0:
        rng = rng or np.random.default_rng(0)
        x += float(rng.uniform(-xy_jitter, xy_jitter))
        z += float(rng.uniform(-xy_jitter, xy_jitter))

    return [x, top_y + drop_height, z]


def settle_object_with_physics(
    scene_handle: str,
    object_template_handle: str,
    start_position: Sequence[float],
    scale: Optional[float] = None,
    dt: float = 1.0 / 60.0,
    max_steps: int = 240,
    settle_lin_vel_eps: float = 0.05,
    settle_ang_vel_eps: float = 0.2,
    settle_steps_required: int = 10,
) -> SettledPlacement:
    """Load scene, drop an object, step physics until it settles.

    Requires Habitat-Sim built with Bullet. Imports habitat_sim lazily.
    """

    try:
        import magnum as mn
        import habitat_sim
        from habitat_sim.utils.common import quat_from_magnum
        from quaternion import as_float_array
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Habitat-Sim + magnum are required for physics placement. "
            "Ensure habitat-sim is installed with Bullet support."
        ) from e

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_handle
    sim_cfg.enable_physics = True

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

    sim = habitat_sim.Simulator(cfg)
    try:
        obj_tmpl_mgr = sim.get_object_template_manager()
        rigid_mgr = sim.get_rigid_object_manager()

        # Load template config.
        template_ids = None
        if hasattr(obj_tmpl_mgr, "load_configs"):
            template_ids = obj_tmpl_mgr.load_configs(object_template_handle)
        if template_ids:
            template_id = template_ids[0]
            obj = rigid_mgr.add_object_by_template_id(template_id)
        else:
            # Fallback for older APIs
            obj = rigid_mgr.add_object_by_template_handle(object_template_handle)

        obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        obj.translation = mn.Vector3(*[float(x) for x in start_position])
        if scale is not None:
            obj.scale = mn.Vector3(scale, scale, scale)

        settled_count = 0
        steps = 0
        for steps in range(1, max_steps + 1):
            sim.step_physics(dt)
            lv = getattr(obj, "linear_velocity", None)
            av = getattr(obj, "angular_velocity", None)
            if lv is None or av is None:
                continue
            if float(np.linalg.norm(lv)) < settle_lin_vel_eps and float(np.linalg.norm(av)) < settle_ang_vel_eps:
                settled_count += 1
            else:
                settled_count = 0
            if settled_count >= settle_steps_required:
                break

        # Serialize final transform
        pos = [float(obj.translation.x), float(obj.translation.y), float(obj.translation.z)]
        q = quat_from_magnum(obj.rotation)
        # quat_from_magnum returns np.quaternion (w,x,y,z). Convert to xyzw coeffs.
        q_arr = as_float_array(q)  # [w,x,y,z]
        rot_xyzw = [float(q_arr[1]), float(q_arr[2]), float(q_arr[3]), float(q_arr[0])]

        return SettledPlacement(position=pos, rotation_quat_xyzw=rot_xyzw, settled=True, steps=steps)
    finally:
        sim.close()
