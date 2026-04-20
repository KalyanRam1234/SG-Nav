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
    navmesh_dist: Optional[float] = None  # distance to nearest navigable point


def propose_drop_start(
    support_node: Dict[str, Any],
    drop_height: float = 0.4,
    xy_jitter: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> List[float]:
    """Compute a plausible start position above a support for physics drop.

    After sg_cache._as_np_xyz() heuristic swap, points are already in
    Habitat convention: [X, Y_up(height), Z].
    """

    c = node_centroid_world(support_node)
    aabb = node_aabb_world(support_node)
    if c is None and aabb is None:
        raise ValueError("Support node has no 3D points (pcd_points/bbox_points)")

    if aabb is not None:
        mn, mx = aabb
        # After _as_np_xyz swap: index 0=X, index 1=Y_up(height), index 2=Z
        hab_x = float((mn[0] + mx[0]) * 0.5)
        hab_z = float((mn[2] + mx[2]) * 0.5)
        top_y = float(mx[1])  # max height of the support surface
    else:
        hab_x = float(c[0])
        hab_z = float(c[2])
        top_y = float(c[1])

    if xy_jitter > 0:
        rng = rng or np.random.default_rng(0)
        hab_x += float(rng.uniform(-xy_jitter, xy_jitter))
        hab_z += float(rng.uniform(-xy_jitter, xy_jitter))

    return [hab_x, top_y + drop_height, hab_z]


def _navmesh_distance(sim, position: Sequence[float]) -> float:
    """Horizontal distance from *position* to the closest navigable point."""
    pf = sim.pathfinder
    if not pf.is_loaded:
        return float("nan")
    best = float("inf")
    pos = np.array(position, dtype=float)
    # Sample random navigable points and keep closest
    for _ in range(5000):
        pt = pf.get_random_navigable_point()
        d = float(np.sqrt((pt[0] - pos[0]) ** 2 + (pt[2] - pos[2]) ** 2))
        if d < best:
            best = d
            if d < 0.5:
                break  # close enough
    return best


def check_navmesh_proximity(
    scene_handle: str,
    positions: List[Sequence[float]],
) -> List[float]:
    """Return navmesh distances for multiple positions in a single sim load."""
    try:
        import habitat_sim
    except ImportError:
        return [float("nan")] * len(positions)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_handle
    sim_cfg.enable_physics = False
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    try:
        # Recompute navmesh if not already loaded
        if not sim.pathfinder.is_loaded:
            nav_settings = habitat_sim.NavMeshSettings()
            nav_settings.set_defaults()
            nav_settings.agent_height = 1.5
            nav_settings.agent_radius = 0.1
            nav_settings.agent_max_climb = 0.2
            sim.recompute_navmesh(sim.pathfinder, nav_settings)
        return [_navmesh_distance(sim, p) for p in positions]
    finally:
        sim.close()


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

    If Habitat-Sim was built without Bullet, falls back to geometric
    placement: the object is placed so its bottom sits on the support
    surface (the Y encoded in *start_position* minus *drop_height*).
    """

    try:
        import magnum as mn
        import habitat_sim
        from habitat_sim.utils.common import quat_from_magnum
        from quaternion import as_float_array
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Habitat-Sim + magnum are required for physics placement. "
            "Ensure habitat-sim is installed."
        ) from e

    has_bullet = getattr(habitat_sim, "built_with_bullet", False)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_handle
    sim_cfg.enable_physics = True

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

    sim = habitat_sim.Simulator(cfg)
    try:
        # Recompute navmesh for distance checks
        if not sim.pathfinder.is_loaded:
            nav_settings = habitat_sim.NavMeshSettings()
            nav_settings.set_defaults()
            sim.recompute_navmesh(sim.pathfinder, nav_settings)

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
            obj = rigid_mgr.add_object_by_template_handle(object_template_handle)

        if scale is not None:
            obj.scale = mn.Vector3(scale, scale, scale)

        if not has_bullet:
            # --- Geometric fallback: place bottom of object on the surface ---
            obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            obj.translation = mn.Vector3(0, 0, 0)

            local_bb = obj.root_scene_node.cumulative_bb
            bottom_y = float(local_bb.min.y)  # negative offset from origin to bottom

            # start_position[1] = surface_top + drop_height.
            # Recover surface_top by subtracting the default drop_height (0.4).
            surface_y = float(start_position[1]) - 0.4
            # Place origin so the bottom sits on the surface (+ tiny margin)
            final_y = surface_y - bottom_y + 0.002

            pos = [float(start_position[0]), final_y, float(start_position[2])]
            obj.translation = mn.Vector3(*pos)

            import warnings
            warnings.warn(
                "Bullet physics not available — using geometric placement. "
                f"Object placed at Y={final_y:.4f} (surface={surface_y:.4f}, "
                f"bb_bottom={bottom_y:.4f})",
                stacklevel=2,
            )
            nav_dist = _navmesh_distance(sim, pos)
            return SettledPlacement(
                position=pos,
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
                settled=False,
                steps=0,
                navmesh_dist=nav_dist,
            )

        # --- Bullet physics path ---
        obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        obj.translation = mn.Vector3(*[float(x) for x in start_position])

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

        pos = [float(obj.translation.x), float(obj.translation.y), float(obj.translation.z)]
        q = quat_from_magnum(obj.rotation)
        q_arr = as_float_array(q)  # [w,x,y,z]
        rot_xyzw = [float(q_arr[1]), float(q_arr[2]), float(q_arr[3]), float(q_arr[0])]

        nav_dist = _navmesh_distance(sim, pos)
        return SettledPlacement(position=pos, rotation_quat_xyzw=rot_xyzw, settled=True, steps=steps, navmesh_dist=nav_dist)
    finally:
        sim.close()
