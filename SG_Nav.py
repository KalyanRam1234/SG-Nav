import argparse
import copy
import faulthandler
import json
import math
import os
import signal
import sys
import time
import warnings
from collections import deque
from datetime import datetime

# Suppress HuggingFace tokenizer fork warnings and prevent deadlocks
# when subprocesses are spawned after CLIP/GroundingDINO init.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Flush stdout after every print so that log files show progress in real time
# (Python uses full buffering when stdout is redirected to a file).
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Enable faulthandler so that SIGUSR1 dumps all thread tracebacks to stderr.
# Usage: kill -USR1 <pid>  → see thread stacks in the log's stderr stream.
faulthandler.enable()
faulthandler.register(signal.SIGUSR1)

# Suppress noisy but harmless warnings from transformers/torch that fire
# on every CLIP forward pass (grid_sample align_corners, device deprecation, etc.)
warnings.filterwarnings("ignore", message=".*align_corners.*")
warnings.filterwarnings("ignore", message=".*`device` argument is deprecated.*")
warnings.filterwarnings("ignore", message=".*requires_grad.*")
warnings.filterwarnings("ignore", message=".*resume_download.*")
from matplotlib import colors
import cv2
import numpy as np
import pandas
import skimage
import torch
import habitat

from GLIP.maskrcnn_benchmark.config import cfg as glip_cfg
from GLIP.maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

from pslpython.model import Model as PSLModel
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule

from scenegraph import SceneGraph

import utils.utils_fmm.control_helper as CH
import utils.utils_fmm.pose_utils as pu
from utils.utils_fmm.fmm_planner import FMMPlanner    
from utils.utils_fmm.mapping import Semantic_Mapping
from utils.utils_glip import *
from utils.image_process import (
    add_resized_image,
    add_rectangle,
    add_text,
    add_text_list,
    crop_around_point,
    draw_agent,
    draw_goal,
    line_list
)


class SG_Nav_Agent():
    def __init__(self, task_config, args=None):
        # Common
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        self.config = task_config
        self.args = args
        self.panoramic = []
        self.panoramic_depth = []
        self.device = (
            torch.device("cuda:{}".format(0))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.turn_angles = 0
        self.prev_action = 0
        self.navigate_steps = 0
        self.move_steps = 0
        self.total_steps = 0

        # Need to adjust how act will handle these variables in the case of exploration
        self.found_goal = False
        self.found_goal_times = 0
        
        # This needs
        self.exploration_total_steps = 0
        
        self.distance_threshold = 5
        self.correct_room = False
        self.changing_room = False
        self.changing_room_steps = 0
        self.move_after_new_goal = False
        self.former_check_step = -10
        self.goal_disappear_step = 100
        self.force_change_room = False
        self.current_room_search_step = 0
        self.target_room = ''
        self.current_rooms = []
        self.nav_without_goal_step = 0
        self.former_collide = 0
        self.history_pose = []
        self.visualize_image_list = []
        self.count_episodes = -1
        self.loop_time = 0
        self.last_segment_num = 0
        self.goal_merge_threshold = 0.8
        self.rooms = rooms
        self.rooms_captions = rooms_captions
        self.split = (self.args.split_l >= 0)
        self.metrics = {'distance_to_goal': 0., 'spl': 0., 'softspl': 0.}

        ### ------ init glip model ------ ###
        config_file = "GLIP/configs/pretrain/glip_Swin_L.yaml" 
        weight_file = "GLIP/MODEL/glip_large_model.pth"
        glip_cfg.local_rank = 0
        glip_cfg.num_gpus = 1
        glip_cfg.merge_from_file(config_file) 
        glip_cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
        glip_cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
        self.glip_demo = GLIPDemo(
            glip_cfg,
            min_image_size=800,
            confidence_threshold=0.61,
            show_mask_heatmaps=False
        )

        self.map_size_cm = 4000
        self.resolution = self.map_resolution = 5
        self.camera_horizon = 0
        self.dilation_deg = 0
        self.collision_threshold = 0.08
        self.selem = skimage.morphology.square(1)
        self.explanation = ''
        self.text_node = ''
        self.text_edge = ''
        self.sem_map_module = Semantic_Mapping(self).to(self.device) 
        self.free_map_module = Semantic_Mapping(self, max_height=10,min_height=-150).to(self.device)
        self.room_map_module = Semantic_Mapping(self, max_height=200,min_height=-10, num_cats=9).to(self.device)
        
        self.free_map_module.eval()
        self.free_map_module.set_view_angles(self.camera_horizon)
        self.sem_map_module.eval()
        self.sem_map_module.set_view_angles(self.camera_horizon)
        self.room_map_module.eval()
        self.room_map_module.set_view_angles(self.camera_horizon)

        self.init_map()

        # Adding global variables initialization
        self.init_global_map()
        self.global_sem_map_module = Semantic_Mapping(self).to(self.device)
        self.global_sem_map_module.eval()
        self.global_sem_map_module.set_view_angles(self.camera_horizon)

        self.global_free_map_module = Semantic_Mapping(self, max_height=10,min_height=-150).to(self.device)
        self.global_free_map_module.eval()
        self.global_free_map_module.set_view_angles(self.camera_horizon)

        self.global_room_map_module = Semantic_Mapping(self, max_height=200,min_height=-10, num_cats=9).to(self.device)
        self.global_room_map_module.eval()

        self.camera_matrix = self.free_map_module.camera_matrix
        
        self.goal_idx = {}
        for key in projection:
            self.goal_idx[projection[key]] = categories_21_origin.index(projection[key])
        # Add extended categories that aren't in projection
        for i, cat in enumerate(categories_21_origin):
            if cat not in self.goal_idx:
                self.goal_idx[cat] = i
        self.co_occur_mtx = np.load('tools/obj.npy')
        self.co_occur_mtx -= self.co_occur_mtx.min()
        self.co_occur_mtx /= self.co_occur_mtx.max() 
        self.num_cooccur_objects = int(self.co_occur_mtx.shape[0])
        
        self.co_occur_room_mtx = np.load('tools/room.npy')
        self.co_occur_room_mtx -= self.co_occur_room_mtx.min()
        self.co_occur_room_mtx /= self.co_occur_room_mtx.max()
        
        self.scenegraph = SceneGraph(map_resolution=self.map_resolution, map_size_cm=self.map_size_cm, map_size=self.map_size, camera_matrix=self.camera_matrix, agent=self)

        # Reuse the same GroundingDINO+SAM and CLIP model instances for the
        # global graph — halves GPU memory (~5-8 GB saved) and startup time.
        shared = {
            'mask_generator': self.scenegraph.mask_generator,
            'clip_model':     self.scenegraph.clip_model,
            'clip_processor': self.scenegraph.clip_processor,
        }
        self.global_scenegraph = SceneGraph(map_resolution=self.map_resolution, map_size_cm=self.map_size_cm, map_size=self.map_size, camera_matrix=self.camera_matrix, agent=self, is_global=True, shared_models=shared)

        self.target_objects_list = [
            'cup', 'chair', 'table', 'sofa', 'cabinet', 'plant', 'lamp', 'picture', 'shower',
            'toilet', 'tv_monitor', 'sink', 'bathtub', 'counter', 'fireplace', 'gym_equipment'
        ]

        self.target_object_idx = 0
        self.found_objects = {}  # Track which objects have been found
        self.detected_objects_extended = set()  # Track extended category detections
        
        # LLM escape mode state
        self.escape_action_queue = []
        self.executing_escape = False
        self.escape_attempts = 0
        self.max_escape_attempts = 2
        self.max_escape_actions = 15
        self.action_history = []

        # Frontier teleport state
        self.saved_frontier_positions = []
        self.frontier_teleport_count = 0
        self.max_frontier_teleports = 5

        # DynamicQA object injection state
        self._inject_manifest = None
        self._inject_variant = getattr(args, 'inject_variant', 0)
        self._inject_record_idx = getattr(args, 'inject_record_idx', None)
        self._injected_objects = []  # track injected rigid objects
        if hasattr(args, 'inject_manifest') and args.inject_manifest:
            self._inject_manifest = self._load_inject_manifest(args.inject_manifest)
            print(f"[Inject] Loaded manifest with {len(self._inject_manifest)} records "
                  f"(variant={self._inject_variant}, record_idx={self._inject_record_idx})")

        # This is to adjust the experiment
        self.experiment_name = 'experiment_0'

        if self.split:
            self.experiment_name = self.experiment_name + f'/[{self.args.split_l}:{self.args.split_r}]'

        # Create a timestamped run folder so each run's outputs are isolated
        from datetime import datetime
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_name = os.path.join(self.experiment_name, f'run_{self.run_timestamp}')

        self.visualization_dir = f'data/visualization/{self.experiment_name}/'

        self.toggle_stuck_handling = True
        print('scene graph module init finish!!!')

    # ── DynamicQA Object Injection ──────────────────────────────────────

    @staticmethod
    def _load_inject_manifest(path):
        """Load a JSONL manifest produced by tools.dynamic_qa.build_dataset."""
        records = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def inject_cad_object(self, template_handle, position, rotation_quat=None, scale=None):
        """Inject a rigid CAD object into the live Habitat-Sim scene.

        The object is placed as STATIC so it stays fixed during navigation.
        Returns the Habitat rigid object handle, or None on failure.
        """
        try:
            import magnum as mn
            import habitat_sim
        except ImportError:
            print("[Inject] ERROR: magnum / habitat_sim not importable")
            return None

        try:
            sim = self.simulator._env._sim
            obj_tmpl_mgr = sim.get_object_template_manager()
            rigid_mgr = sim.get_rigid_object_manager()

            template_ids = obj_tmpl_mgr.load_configs(template_handle)
            if not template_ids:
                print(f"[Inject] ERROR: could not load template {template_handle}")
                return None
            template_id = template_ids[0]
            obj = rigid_mgr.add_object_by_template_id(template_id)

            obj.motion_type = habitat_sim.physics.MotionType.STATIC
            obj.translation = mn.Vector3(*[float(v) for v in position])

            if rotation_quat is not None:
                # rotation_quat is [x, y, z, w]
                obj.rotation = mn.Quaternion(
                    mn.Vector3(float(rotation_quat[0]),
                               float(rotation_quat[1]),
                               float(rotation_quat[2])),
                    float(rotation_quat[3])
                )
            if scale is not None:
                s = float(scale)
                obj.scale = mn.Vector3(s, s, s)

            self._injected_objects.append(obj)
            print(f"[Inject] Placed object at {position} (template={template_handle})")
            return obj
        except Exception as e:
            print(f"[Inject] ERROR injecting object: {e}")
            return None

    def _inject_from_manifest(self):
        """Match the current episode's scene to the manifest and inject objects.

        Called from reset() after the episode is loaded.
        """
        if self._inject_manifest is None:
            return

        episode = self.simulator._env.current_episode
        episode_scene = getattr(episode, 'scene_id', '') or ''

        # Determine which manifest records to use
        if self._inject_record_idx is not None:
            # Use a specific record
            idx = int(self._inject_record_idx)
            if idx < len(self._inject_manifest):
                records = [self._inject_manifest[idx]]
            else:
                print(f"[Inject] WARNING: record_idx {idx} out of range "
                      f"(manifest has {len(self._inject_manifest)} records)")
                return
        else:
            # Auto-match by scene_handle
            records = [r for r in self._inject_manifest
                       if self._scene_matches(r.get('scene_handle', ''), episode_scene)]
            if not records:
                print(f"[Inject] No manifest records match scene {episode_scene}")
                return

        variant_idx = int(self._inject_variant)
        for record in records:
            placements = record.get('validPlacements', [])
            if variant_idx >= len(placements):
                print(f"[Inject] WARNING: variant {variant_idx} not in placements "
                      f"(have {len(placements)})")
                continue
            placement = placements[variant_idx]
            template = record.get('template_handle', '')
            position = placement.get('position', [0, 0, 0])
            rotation = placement.get('rotation', None)
            cat = record.get('object_category', 'unknown')
            print(f"[Inject] Injecting '{cat}' variant={variant_idx} into scene")
            self.inject_cad_object(template, position, rotation)

    @staticmethod
    def _scene_matches(manifest_scene, episode_scene):
        """Check if a manifest scene_handle matches the episode's scene_id.

        Compares the scene directory name (e.g. '2azQ1b91cZZ') since paths
        may differ between generation and runtime environments.
        """
        if not manifest_scene or not episode_scene:
            return False
        # Extract the scene directory component (e.g. '2azQ1b91cZZ')
        m_parts = manifest_scene.replace('\\', '/').split('/')
        e_parts = episode_scene.replace('\\', '/').split('/')
        # Compare last directory-like component (before .glb)
        m_key = [p for p in m_parts if p and not p.endswith('.glb')]
        e_key = [p for p in e_parts if p and not p.endswith('.glb')]
        if m_key and e_key:
            return m_key[-1] == e_key[-1]
        return manifest_scene in episode_scene or episode_scene in manifest_scene

    def add_predicates(self, model):
        predicate = Predicate('IsNearObj', closed = True, size = 2)
        model.add_predicate(predicate)
        predicate = Predicate('ObjCooccur', closed = True, size = 1)
        model.add_predicate(predicate)
        predicate = Predicate('IsNearRoom', closed = True, size = 2)
        model.add_predicate(predicate)
        predicate = Predicate('RoomCooccur', closed = True, size = 1)
        model.add_predicate(predicate)
        predicate = Predicate('Choose', closed = False, size = 1)
        model.add_predicate(predicate)
        predicate = Predicate('ShortDist', closed = True, size = 1)
        model.add_predicate(predicate)
        
    def add_rules(self, model):
        model.add_rule(Rule('2: ObjCooccur(O) & IsNearObj(O,F)  -> Choose(F)^2'))
        model.add_rule(Rule('2: !ObjCooccur(O) & IsNearObj(O,F) -> !Choose(F)^2'))
        model.add_rule(Rule('2: RoomCooccur(R) & IsNearRoom(R,F) -> Choose(F)^2'))
        model.add_rule(Rule('2: !RoomCooccur(R) & IsNearRoom(R,F) -> !Choose(F)^2'))
        model.add_rule(Rule('2: ShortDist(F) -> Choose(F)^2'))
        model.add_rule(Rule('Choose(+F) = 1 .'))
    
    def get_current_target_object(self):
        """Get the current target object to search for"""
        if hasattr(self.scenegraph, 'obj_goal') and self.scenegraph.obj_goal not in self.target_objects_list:
            self.target_objects_list = [self.scenegraph.obj_goal] + self.target_objects_list
        
        # Initialize found_objects dict if needed
        for obj in self.target_objects_list:
            if obj not in self.found_objects:
                self.found_objects[obj] = False
        
        if self.target_object_idx >= len(self.target_objects_list):
            return None
        
        current_target = self.target_objects_list[self.target_object_idx]
        if not self.found_objects[current_target]:
            self.obj_goal = current_target
            self.obj_goal_sg = current_target
            return current_target
        
        return None

    def reset(self):
        self.navigate_steps = 0
        self.turn_angles = 0
        self.move_steps = 0
        self.total_steps = 0
        self.current_room_search_step = 0
        self.found_goal = False
        self.found_goal_times = 0
        self.correct_room = False
        self.changing_room = False
        self.goal_loc = None
        self.changing_room_steps = 0
        self.move_after_new_goal = False
        self.former_check_step = -10
        self.goal_disappear_step = 100
        self.prev_action = 0
        self.former_collide = 0
        self.goal_gps = np.array([0.,0.])
        self.possible_goal_temp_gps = np.array([0.,0.])
        self.last_gps = np.array([11100.,11100.])
        self.has_panarama = False
        self.init_map()
        self.last_loc = self.full_pose
        self.panoramic = []
        self.panoramic_depth = []
        self.current_rooms = []
        self.dist_to_frontier_goal = 10
        self.first_fbe = True
        self.goal_map = np.zeros(self.full_map.shape[-2:])
        self.found_possible_goal = False
        self.history_pose = []
        self.visualize_image_list = []
        self.count_episodes = self.count_episodes + 1
        self.loop_time = 0
        self.last_segment_num = 0
        self.metrics = {'distance_to_goal': 0., 'spl': 0., 'softspl': 0.}
        self.explanation = ''
        self.text_node = ''
        self.text_edge = ''
        self.obj_goal = self.simulator._env.current_episode.object_category
        self.obj_goal_sg = self.simulator._env.current_episode.object_category
        if self.obj_goal == 'gym_equipment':
            self.obj_goal_sg = 'treadmill. fitness equipment.'
        elif self.obj_goal == 'chest_of_drawers':
            self.obj_goal_sg = 'drawers'
        elif self.obj_goal == 'tv_monitor':
            self.obj_goal_sg = 'tv'

        episode = self.simulator._env.current_episode
        total_episodes = len(self.simulator._env.episodes)
        print(f"\n{'='*80}")
        print(f"[Episode Reset] Episode {self.count_episodes}/{total_episodes} | "
              f"ID: {episode.episode_id} | Scene: {episode.scene_id.split('/')[-2]} | "
              f"Goal: {self.obj_goal}")
        print(f"{'='*80}\n")
        self.current_obj_predictions = []
        self.obj_locations = [[] for i in range(self.num_cooccur_objects)]
        self.not_move_steps = 0
        self.move_since_random = 0
        self.using_random_goal = False
        self.fronter_this_ex = 0
        self.random_this_ex = 0
        self.last_location = np.array([0.,0.])
        self.detected_objects_extended = set()
        self.escape_action_queue = []
        self.executing_escape = False
        self.escape_attempts = 0
        self.action_history = []
        self.saved_frontier_positions = []
        self.frontier_teleport_count = 0

        self.scenegraph.reset()

        # Inject CAD objects from DynamicQA manifest (if configured)
        self._injected_objects.clear()
        self._inject_from_manifest()

        # Override navigation goal to the injected object's category
        if self._inject_manifest and self._injected_objects:
            rec_idx = self._inject_record_idx if self._inject_record_idx is not None else 0
            if rec_idx < len(self._inject_manifest):
                injected_cad = self._inject_manifest[rec_idx].get('object_category', '')
                if injected_cad:
                    self.obj_goal = injected_cad
                    self.obj_goal_sg = injected_cad
                    if injected_cad not in self.target_objects_list:
                        self.target_objects_list = [injected_cad] + self.target_objects_list
                        self.target_object_idx = 0
                    print(f"[Inject] Navigation goal overridden to '{injected_cad}'")
        
    def reset_local_scenegraph(self):
        """Reset only the local scene graph after finding a goal object"""
        self.total_steps = 0
        self.navigate_steps = 0
        self.move_steps = 0
        self.found_goal = False
        self.found_goal_times = 0
        self.first_fbe = True
        self.goal_map = np.zeros(self.full_map.shape[-2:])

        # should handle setting the goal for the next object
        if self.target_object_idx is not None and self.target_object_idx < len(self.target_objects_list): 
            self.obj_goal = self.target_objects_list[self.target_object_idx]
            self.obj_goal_sg = self.target_objects_list[self.target_object_idx]
        if self.obj_goal == 'gym_equipment':
            self.obj_goal_sg = 'treadmill. fitness equipment.'
        elif self.obj_goal == 'chest_of_drawers':
            self.obj_goal_sg = 'drawers'
        elif self.obj_goal == 'tv_monitor':
            self.obj_goal_sg = 'tv'

        self.goal_loc = None

        # Need to create one to initialize and set all the global scene variables of map
        # so that local scene graph can be reset, but the global one is persistent
        # Mainly need to set the fbe_free_map correctly.
        # This method will be used to reset the key variables of the scene, from which the next goal detection will begin.
        self.init_map()
        self.prev_action = 0
        self.former_collide = 0
        self.goal_gps = np.array([0.,0.])
        self.possible_goal_temp_gps = np.array([0.,0.])
        self.last_loc = self.full_pose
        self.panoramic = []
        self.panoramic_depth = []
        self.current_rooms = []
        self.dist_to_frontier_goal = 10
        self.found_possible_goal = False
        self.history_pose = []
        # Re-add current pose to history after reset (for visualization and tracking)
        if hasattr(self, 'full_pose'):
            self.history_pose.append(self.full_pose.cpu().detach().clone())
            print(f"[Reset] Re-initialized history_pose with current agent pose")
        self.loop_time = 0

        # need custom metrics for global scene graph
        self.metrics = {'distance_to_goal': 0., 'spl': 0., 'softspl': 0.}
        self.current_obj_predictions = []
        self.obj_locations = [[] for i in range(self.num_cooccur_objects)]
        self.not_move_steps = 0
        self.move_since_random = 0
        self.using_random_goal = False
        self.last_location = np.array([0.,0.])
        self.current_stuck_steps = 0
        self.total_stuck_steps = 0
        self.explanation = ''
        self.text_node = ''
        self.text_edge = ''
        self.escape_action_queue = []
        self.executing_escape = False
        self.escape_attempts = 0
        self.frontier_teleport_count = 0
        
        # IMPORTANT: Reset the local scene graph nodes and edges (but NOT global)
        print(f"[Reset] Resetting LOCAL scene graph (nodes and edges)")
        self.scenegraph.reset()
        print(f"[Reset] LOCAL scene graph reset - Nodes: {len(self.scenegraph.nodes)}, Edges: {len(self.scenegraph.get_edges())}")

    def detect_objects(self, observations):
        # This variable only changes based on inference of current scene
        self.current_obj_predictions = self.glip_demo.inference(observations["rgb"][:,:,[2,1,0]], object_captions) # GLIP object detection, time cosuming
        new_labels = self.get_glip_real_label(self.current_obj_predictions) # transfer int labels to string labels
        self.current_obj_predictions.add_field("labels", new_labels)

        obj_labels = self.current_obj_predictions.get_field("labels")
        obj_scores = self.current_obj_predictions.get_field("scores")
        detected_summary = {}
        for j, label in enumerate(obj_labels):
            if label not in detected_summary:
                detected_summary[label] = 0
            detected_summary[label] += 1
        print(f"[DetectObjects] GLIP detected {len(obj_labels)} objects: {detected_summary}")

        shortest_distance = 120
        shortest_distance_angle = 0
        goal_bbox = []
        for j, label in enumerate(obj_labels):
            if self.obj_goal in label:
                goal_bbox.append(self.current_obj_predictions.bbox[j])
            elif self.obj_goal == 'gym_equipment' and (label in ['treadmill', 'exercise machine']):
                goal_bbox.append(self.current_obj_predictions.bbox[j])
        
        for j, label in enumerate(obj_labels):
            # Track objects in original 21 categories for navigation scoring
            if label in categories_21_origin:
                confidence = self.current_obj_predictions.get_field("scores")[j]
                bbox = self.current_obj_predictions.bbox[j].to(torch.int64)
                center_point = (bbox[:2] + bbox[2:]) // 2
                temp_direction = (center_point[0] - 320) * 79 / 640

                # comes from observations, where act() will first initialize this variable.
                temp_distance = self.depth[center_point[1],center_point[0],0]
                if temp_distance >= self.distance_threshold:
                    continue

                obj_gps = self.get_goal_gps(observations, temp_direction, temp_distance)
                x = int(self.map_size_cm/10-obj_gps[1]*100/self.resolution)
                y = int(self.map_size_cm/10+obj_gps[0]*100/self.resolution)
                idx = categories_21_origin.index(label)
                if idx < self.num_cooccur_objects:
                    self.obj_locations[idx].append([confidence, x, y])
                else:
                    self.detected_objects_extended.add(label)
            elif label in categories_extended:
                # Track extended category detections for richer scene graph
                confidence = self.current_obj_predictions.get_field("scores")[j]
                self.detected_objects_extended.add(label)
        
        # it is to be noted that the scenegraph.obj_goal is what will change,  this is for smaller object detection
        if self.scenegraph.obj_goal in self.scenegraph.small_objects:
            self.segment_num = len(self.scenegraph.segment2d_results)
            goal_mask = []
            # comes from scenegraph update
            if self.segment_num > self.last_segment_num:
                self.last_segment_num = self.segment_num
                segment2d_result = self.scenegraph.segment2d_results[-1]
                indices = []
                for index, element in enumerate(segment2d_result['caption']):
                    if self.obj_goal_sg in element.split(' '):
                        for node in self.scenegraph.nodes:
                            if node.is_goal_node and node.object['image_idx'][-1] == len(self.scenegraph.segment2d_results) - 1 and node.object['mask_idx'][-1] == index:
                                indices.append(index)
                goal_mask = [segment2d_result['mask'][index] for index in indices]
            if len(goal_mask) > 0:
                possible_goal_detected_before = copy.deepcopy(self.found_possible_goal)
                for mask in goal_mask:
                    center_point = torch.tensor(np.argwhere(mask).mean(axis=0).astype(int))
                    center_point = torch.tensor([center_point[1], center_point[0]])
                    temp_direction = (center_point[0] - 320) * 79 / 640
                    # self.depth comes from the act() observations
                    temp_distance = self.depth[center_point[1],center_point[0],0]
                    k = 0
                    pos_neg = 1
                    while temp_distance >= 100 and 0<center_point[1]+int(pos_neg*k)<479 and 0<center_point[0]+int(pos_neg*k)<639:
                        pos_neg *= -1
                        k += 0.5
                        temp_distance = max(self.depth[center_point[1]+int(pos_neg*k),center_point[0],0],
                        self.depth[center_point[1],center_point[0]+int(pos_neg*k),0])
                        
                    if temp_distance >= self.distance_threshold:
                        self.found_possible_goal = True
                    else:
                        # This is where the goal is marked as found, then in the action, it returns a stop or reset of the agent occurs
                        if self.found_goal:
                            if temp_distance < self.distance_threshold:
                                self.found_goal_times = self.found_goal_times + 1
                        self.found_goal = True
                        self.found_possible_goal = False
                    
                    ## select the closest goal
                    direction = temp_direction
                    distance = temp_distance
                    if distance < shortest_distance:
                        shortest_distance = distance
                        shortest_distance_angle = direction
                
                if self.found_goal:
                    self.goal_gps = self.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
                elif not possible_goal_detected_before:
                    # if detected a long goal before, then don't change it until see a goal within 5 meters
                    self.possible_goal_temp_gps = self.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
            else:
                # means goal mask isn't found in current observation
                if self.found_goal:
                    self.found_goal = False
                    self.found_goal_times = 0
            return
        else:
            # This is to handle detection of larger objects
            if len(goal_bbox) > 0:
                possible_goal_detected_before = copy.deepcopy(self.found_possible_goal)
                stacked_goal_bbox = torch.stack(goal_bbox)
                for box in stacked_goal_bbox:
                    box = box.to(torch.int64)
                    center_point = (box[:2] + box[2:]) // 2
                    temp_direction = (center_point[0] - 320) * 79 / 640
                    temp_distance = self.depth[center_point[1],center_point[0],0]
                    goal_gps = self.get_goal_gps(observations, temp_direction, temp_distance)
                    k = 0
                    pos_neg = 1
                    while temp_distance >= 100 and 0<center_point[1]+int(pos_neg*k)<479 and 0<center_point[0]+int(pos_neg*k)<639:
                        pos_neg *= -1
                        k += 0.5
                        temp_distance = max(self.depth[center_point[1]+int(pos_neg*k),center_point[0],0],
                        self.depth[center_point[1],center_point[0]+int(pos_neg*k),0])
                        
                    if temp_distance >= self.distance_threshold:
                        self.found_possible_goal = True
                    else:
                        thres = int(self.goal_merge_threshold * 100 / self.map_resolution)
                        if 0 <= int(self.map_size_cm/10+goal_gps[0]*100/self.resolution) < self.map_size and 0 <= int(self.map_size_cm/10+goal_gps[1]*100/self.resolution) < self.map_size:
                            goal_gps_map_local = self.goal_gps_map[max(int(self.map_size_cm/10+goal_gps[1]*100/self.resolution) - thres, 0):min(int(self.map_size_cm/10+goal_gps[1]*100/self.resolution) + thres, self.map_size - 1), max(int(self.map_size_cm/10+goal_gps[0]*100/self.resolution) - thres, 0):min(int(self.map_size_cm/10+goal_gps[0]*100/self.resolution) + thres, self.map_size - 1)]
                            if goal_gps_map_local.max() > 0:
                                goal_gps_map_local[np.where(goal_gps_map_local == goal_gps_map_local.max())[0][0], np.where(goal_gps_map_local == goal_gps_map_local.max())[1][0]] = goal_gps_map_local[np.where(goal_gps_map_local == goal_gps_map_local.max())[0][0], np.where(goal_gps_map_local == goal_gps_map_local.max())[1][0]] + 1
                            else:
                                self.goal_gps_map[min(max(int(self.map_size_cm/10+goal_gps[1]*100/self.resolution), 0), self.map_size), min(max(int(self.map_size_cm/10+goal_gps[0]*100/self.resolution), 0), self.map_size)] = 1
                        self.found_possible_goal = False
                    
                    direction = temp_direction
                    distance = temp_distance
                    if distance < shortest_distance:
                        shortest_distance = distance
                        shortest_distance_angle = direction
                
                self.found_goal_times = self.goal_gps_map.max()
                if self.found_goal_times >= self.scenegraph.cfg.obj_min_detections:
                    self.found_goal = True

                if self.found_goal:
                    self.goal_gps = np.flip(np.array(np.where(self.goal_gps_map == self.goal_gps_map.max()))[:, 0])
                    self.goal_gps = (self.goal_gps - self.map_size_cm / 10) / 100 * self.resolution
                elif not possible_goal_detected_before:
                    self.possible_goal_temp_gps = self.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
            return

    def get_global_metrics(self):
        """Calculate and return global scene graph metrics"""
        # Calculate global free map coverage percentage
        if hasattr(self, 'global_fbe_free_map') and self.global_fbe_free_map is not None:
            # Convert tensor to numpy/float for safe computation
            fbe_map = self.global_fbe_free_map
            if isinstance(fbe_map, torch.Tensor):
                fbe_map = fbe_map.cpu().numpy()
            fbe_free_coverage = (fbe_map > 0.5).sum() / fbe_map.size * 100
        else:
            fbe_free_coverage = 0.0
        
        # Get number of nodes and edges in global scene graph
        if hasattr(self, 'global_scenegraph'):
            num_nodes = len(self.global_scenegraph.nodes)
            num_edges = len(self.global_scenegraph.get_edges())
        else:
            num_nodes = 0
            num_edges = 0
        
        return {
            'global_fbe_free_coverage': fbe_free_coverage,
            'num_nodes': num_nodes,
            'num_edges': num_edges
        }
    
    def print_metrics(self):
        """Print verbose metrics at each step"""
        metrics = self.get_global_metrics()
        print(f"\n[Step {self.total_steps:4d}] [Nav Step {self.navigate_steps:4d}] "
              f"Global FBE Free Coverage: {metrics['global_fbe_free_coverage']:6.2f}% | "
              f"Nodes: {metrics['num_nodes']:3d} | "
              f"Edges: {metrics['num_edges']:3d} | "
              f"Target: {self.obj_goal}")
        
        # Log unique objects detected in scene graphs
        local_objects = set(node.caption for node in self.scenegraph.nodes)
        global_objects = set(node.caption for node in self.global_scenegraph.nodes)
        all_detected = local_objects | global_objects | self.detected_objects_extended
        print(f"[Objects] Local SG unique: {sorted(local_objects)} ({len(local_objects)})")
        print(f"[Objects] Global SG unique: {sorted(global_objects)} ({len(global_objects)})")
        if self.detected_objects_extended:
            print(f"[Objects] Extended detections: {sorted(self.detected_objects_extended)} ({len(self.detected_objects_extended)})")
        print(f"[Objects] Total unique objects detected: {sorted(all_detected)} ({len(all_detected)})")

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
        
        # 3. Global scene graph
        lines.append(f"\n=== SCENE GRAPH ===")
        
        if hasattr(self.global_scenegraph, 'room_nodes') and self.global_scenegraph.room_nodes:
            room_names = [r.caption for r in self.global_scenegraph.room_nodes if r.nodes]
            if room_names:
                lines.append(f"Rooms with objects: {', '.join(room_names)}")
        
        global_objects = sorted(set(node.caption for node in self.global_scenegraph.nodes))
        local_objects = sorted(set(node.caption for node in self.scenegraph.nodes))
        lines.append(f"Objects seen globally: {', '.join(global_objects) if global_objects else 'none'}")
        lines.append(f"Objects seen locally: {', '.join(local_objects) if local_objects else 'none'}")
        
        edges = self.global_scenegraph.get_edges()
        if edges:
            edge_texts = [e.text_with_snapshot() for e in edges[:20]]
            lines.append(f"Spatial relationships: {'; '.join(edge_texts)}")
            snapshot_count = sum(1 for e in edges if e.has_snapshot)
            if snapshot_count > 0:
                lines.append(f"({snapshot_count}/{len(edges)} edges have memory snapshots)")
        
        # 4. Spatial map summary
        lines.append(f"\n=== SPATIAL MAP ===")
        lines.append(self._get_spatial_map_summary())
        
        return '\n'.join(lines)

    def _get_spatial_map_summary(self):
        """Summarize spatial maps for the VLM prompt."""
        lines = []
        
        if hasattr(self, 'global_fbe_free_map') and self.global_fbe_free_map is not None:
            free_map = self.global_fbe_free_map[0, 0].cpu().numpy()
            total_cells = free_map.size
            free_cells = (free_map > 0).sum()
            lines.append(f"Explored area: {free_cells}/{total_cells} cells "
                         f"({100*free_cells/total_cells:.1f}%)")
        
        if hasattr(self, 'global_collision_map'):
            collision_cells = (self.global_collision_map > 0).sum()
            lines.append(f"Collision cells: {collision_cells}")
        
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
        
        if self.current_rooms:
            lines.append(f"Current room(s): {', '.join(self.current_rooms)}")
        
        return '\n'.join(lines)

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

    def _parse_escape_actions(self, vlm_response):
        """Parse VLM response into a list of integer actions."""
        action_map = {
            'move_forward': 1,
            'turn_left': 2,
            'turn_right': 3,
        }
        
        actions = []
        response_clean = vlm_response.strip().lower()
        response_clean = response_clean.replace('`', '').replace('*', '')
        tokens = [t.strip().strip('.') for t in response_clean.split(',')]
        
        for token in tokens:
            if token in action_map:
                actions.append(action_map[token])
            elif 'forward' in token:
                actions.append(1)
            elif 'left' in token:
                actions.append(2)
            elif 'right' in token:
                actions.append(3)
        
        actions = actions[:self.max_escape_actions]
        return actions

    def llm_plan_escape(self, observations):
        """Use VLM to plan an escape sequence when the robot is stuck."""
        print(f"[LLM Escape] Planning escape (attempt {self.escape_attempts + 1}/{self.max_escape_attempts})")
        
        prompt = self._build_escape_prompt()
        print(f"[LLM Escape] Prompt:\n{prompt}")
        
        try:
            from PIL import Image
            rgb_image = Image.fromarray(observations["rgb"])
            response = self.scenegraph.get_vlm_response(prompt=prompt, image=rgb_image)
            print(f"[LLM Escape] VLM response: {response}")
        except Exception as e:
            print(f"[LLM Escape] VLM call failed: {e}")
            self.escape_attempts += 1
            return False
        
        if not response:
            print(f"[LLM Escape] Empty VLM response")
            self.escape_attempts += 1
            return False
        
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

    # Observations are the sensor informations       
    def act(self, observations):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            free_device, total_device = _get_free_and_total_gpu_memory()
            free_device_gb = free_device / 1024**3
            total_device_gb = total_device / 1024**3
            used_by_others = total_device_gb - free_device_gb - reserved
            print(f"[GPU] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Free(device): {free_device_gb:.2f} GB | Total: {total_device_gb:.2f} GB | Others: {used_by_others:.2f} GB | Step: {self.total_steps}")
        if self.total_steps >= 300:
            print(f"[Act] Max steps reached: {self.total_steps}")
            self.found_objects[self.obj_goal] = False
            self.target_object_idx += 1
            
            # Transition to next goal if available
            if self.target_object_idx < len(self.target_objects_list):
                print(f"[Act] Transitioning to next goal: {self.target_objects_list[self.target_object_idx]}")
                # Save video for the object that was just found BEFORE resetting
                if self.args.visualize:
                    print(f"[Act] Saving video for object: {self.obj_goal}")
                    self.save_video_for_object(self.obj_goal)
                self.save_global_scenegraph(tag=f"goal_{self.obj_goal}_{self.target_object_idx}_moving_on_post_300_steps")
                self.reset_local_scenegraph()
            else:
                # All objects found
                print(f"[Act] All target objects found! Episode complete.")
                # Save final video for last object
                if self.args.visualize:
                    self.save_video()
                self.save_global_scenegraph(tag="episode_complete")
                return {"action": 0}
        
        print(f"\n[Episode info] total_episodes = {len(self.simulator._env.episodes)}|")

        self.total_steps += 1
        self.exploration_total_steps +=1

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

        # Determine current target object
        print(f"\n{'='*100}")
        print(f"[Act] Step {self.total_steps} - Determining current target object...")
        current_target = self.get_current_target_object()
        if current_target is None:
            # All objects found, episode complete
            print(f"[Act] All target objects found! Episode complete.")
            self.save_video()
            self.save_global_scenegraph(tag="episode_complete")
            return {"action": 0}
        print(f"[Act] Current target object: {current_target}")
        
        if self.navigate_steps == 0:
            print(f"[Act] Initialize goal probabilities for: {self.obj_goal}")
            if self.obj_goal in self.goal_idx and self.goal_idx[self.obj_goal] < self.num_cooccur_objects:
                self.prob_array_room = self.co_occur_room_mtx[self.goal_idx[self.obj_goal]]
                self.prob_array_obj = self.co_occur_mtx[self.goal_idx[self.obj_goal]]
            else:
                # Unknown goal category — use uniform prior (no co-occurrence bias)
                print(f"[Act] WARNING: '{self.obj_goal}' not in co-occurrence matrices, using uniform prior")
                self.prob_array_room = np.ones(self.co_occur_room_mtx.shape[1]) / self.co_occur_room_mtx.shape[1]
                self.prob_array_obj = np.ones(self.num_cooccur_objects) / float(self.num_cooccur_objects)

        _step_t0 = time.perf_counter()

        print(f"[Act] Processing observations - depth clipping...")
        observations["depth"][observations["depth"]==0.5] = 100 # don't construct unprecise map with distance less than 0.5 m
        self.depth = observations["depth"]
        self.rgb = observations["rgb"][:,:,[2,1,0]]
        self.rgb_visualization = observations["rgb"]

        _t = time.perf_counter()
        print(f"[Act] Updating LOCAL scene graph...")
        self.scenegraph.set_agent(self)
        self.scenegraph.set_navigate_steps(self.navigate_steps)
        self.scenegraph.set_obj_goal(self.obj_goal, self.obj_goal_sg)
        self.scenegraph.set_room_map(self.room_map)
        self.scenegraph.set_fbe_free_map(self.fbe_free_map)
        self.scenegraph.set_observations(observations)
        self.scenegraph.set_full_map(self.full_map)
        self.scenegraph.set_full_pose(self.full_pose)
        self.scenegraph.update_scenegraph()
        local_node_captions = [n.caption for n in self.scenegraph.nodes]
        _local_sg_ms = (time.perf_counter() - _t) * 1000
        print(f"[Act] LOCAL scene graph updated - Nodes: {len(self.scenegraph.nodes)}, Edges: {len(self.scenegraph.get_edges())} [{_local_sg_ms:.0f}ms]")
        print(f"[Act] LOCAL nodes: {local_node_captions}")
        
        # Update GLOBAL scene graph (persistent)
        _t = time.perf_counter()
        print(f"[Act] Updating GLOBAL scene graph...")
        self.global_scenegraph.set_agent(self)
        self.global_scenegraph.set_navigate_steps(self.navigate_steps)
        self.global_scenegraph.set_obj_goal(self.obj_goal, self.obj_goal_sg)
        self.global_scenegraph.set_room_map(self.global_room_map)
        self.global_scenegraph.set_fbe_free_map(self.global_fbe_free_map)
        self.global_scenegraph.set_observations(observations)
        self.global_scenegraph.set_full_map(self.global_full_map)
        self.global_scenegraph.set_full_pose(self.full_pose)
        self.global_scenegraph.update_scenegraph()
        global_node_captions = [n.caption for n in self.global_scenegraph.nodes]
        _global_sg_ms = (time.perf_counter() - _t) * 1000
        print(f"[Act] GLOBAL scene graph updated - Nodes: {len(self.global_scenegraph.nodes)}, Edges: {len(self.global_scenegraph.get_edges())} [{_global_sg_ms:.0f}ms]")
        print(f"[Act] GLOBAL nodes: {global_node_captions}")

        # Need Global ones too, done
        _t = time.perf_counter()
        print(f"[Act] Updating local maps...")
        self.update_map(observations)
        self.update_free_map(observations)
        
        print(f"[Act] Updating global maps...")
        self.update_global_free_map(observations)
        self.update_global_map(observations)
        _maps_ms = (time.perf_counter() - _t) * 1000
        print(f"[Act] Maps updated successfully [{_maps_ms:.0f}ms]")

        if self.total_steps == 1:
            print(f"[Act] Step 1: Setting view angle to 30 degrees (initial lookup)")
            self.sem_map_module.set_view_angles(30)
            self.global_sem_map_module.set_view_angles(30)

            self.free_map_module.set_view_angles(30)
            self.global_free_map_module.set_view_angles(30)
            self.print_metrics()
            return {"action": 5}
        elif self.total_steps <= 7:
            print(f"[Act] Steps 2-7: Panoramic rotation (right)")
            return {"action": 6}
        elif self.total_steps == 8:
            print(f"[Act] Step 8: Setting view angle to 60 degrees (upward)")
            self.sem_map_module.set_view_angles(60)
            self.global_sem_map_module.set_view_angles(60)

            self.free_map_module.set_view_angles(60)
            self.global_free_map_module.set_view_angles(60)
            return {"action": 5}
        elif self.total_steps <= 14:
            print(f"[Act] Steps 9-14: Panoramic rotation (right) at high angle")
            return {"action": 6}
        elif self.total_steps <= 15:
            print(f"[Act] Step 15: Setting view angle back to 30 degrees")
            self.sem_map_module.set_view_angles(30)
            self.global_sem_map_module.set_view_angles(30)

            self.free_map_module.set_view_angles(30)
            self.global_free_map_module.set_view_angles(30)
            return {"action": 4}
        elif self.total_steps <= 16:
            print(f"[Act] Step 16: Setting view angle to 0 degrees (forward)")
            self.sem_map_module.set_view_angles(0)
            self.global_sem_map_module.set_view_angles(0)

            self.free_map_module.set_view_angles(0)
            self.global_free_map_module.set_view_angles(0)
            return {"action": 4}
        if self.total_steps <= 22 and not self.found_goal:
            print(f"[Act] Steps 1-22: Initial panoramic exploration")
            self.panoramic.append(observations["rgb"][:,:,[2,1,0]])
            self.panoramic_depth.append(observations["depth"])
            # This gets triggered regularly
            _t = time.perf_counter()
            print(f"[Act] Detecting objects in current view...")
            self.detect_objects(observations)
            _detect_ms = (time.perf_counter() - _t) * 1000
            print(f"[Act] Object detection done [{_detect_ms:.0f}ms]")
            _t = time.perf_counter()
            print(f"[Act] Detecting room layout...")
            room_detection_result = self.glip_demo.inference(observations["rgb"][:,:,[2,1,0]], rooms_captions)
            self.update_room_map(observations, room_detection_result)
            print(f"[Act] Updating LOCAL room map")

            # adding global scene graph room information
            print(f"[Act] Updating GLOBAL room map")
            self.update_global_room_map(observations, room_detection_result)
            _room_ms = (time.perf_counter() - _t) * 1000
            print(f"[Act] Room detection + map update done [{_room_ms:.0f}ms]")

            if not self.found_goal: # if found a goal, directly go to it
                print(f"[Act] Goal not found yet, continuing panoramic rotation")
                self.print_metrics()
                return {"action": 6}
                    
        if np.linalg.norm(observations["gps"] - self.last_gps) >= 0.05:
            self.move_steps += 1
            self.not_move_steps = 0
            if self.using_random_goal:
                self.move_since_random += 1
            # Save waypoint every 20 movement steps for frontier teleport
            if self.move_steps % 20 == 0:
                self._save_agent_waypoint()
            print(f"[Act] Agent moved - Move steps: {self.move_steps}")
        else:
            self.not_move_steps += 1
            print(f"[Act] Agent stationary - Not move steps: {self.not_move_steps}")
            
        self.last_gps = observations["gps"]
        
        # only doing once is fine since they refer to same agent
        _t = time.perf_counter()
        print(f"[Act] Running scene graph perception...")
        self.scenegraph.perception()
        _percep_ms = (time.perf_counter() - _t) * 1000
        print(f"[Act] Perception done [{_percep_ms:.0f}ms]")
          
        self.history_pose.append(self.full_pose.cpu().detach().clone())
        print(f"[Act] Recorded agent pose - History length: {len(self.history_pose)}")
        
        print(f"[Act] Computing traversible map from agent pose...")
        input_pose = np.zeros(7)
        input_pose[:3] = self.full_pose.cpu().numpy()
        input_pose[1] = self.map_size_cm/100 - input_pose[1]
        input_pose[2] = -input_pose[2]
        input_pose[4] = self.full_map.shape[-2]
        input_pose[6] = self.full_map.shape[-1]
        traversible, cur_start, cur_start_o = self.get_traversible(self.full_map.cpu().numpy()[0,0,::-1], input_pose)
        print(f"[Act] Traversible map computed - Agent position: ({cur_start[0]:.2f}, {cur_start[1]:.2f}), Orientation: {cur_start_o:.2f}°")
        
        if self.found_goal: 
            print(f"[Act] GOAL FOUND! {self.obj_goal}")
            self.found_objects[self.obj_goal] = True
            self.target_object_idx += 1
            
            # Transition to next goal if available
            if self.target_object_idx < len(self.target_objects_list):
                print(f"[Act] Transitioning to next goal: {self.target_objects_list[self.target_object_idx]}")
                # Save video for the object that was just found BEFORE resetting
                if self.args.visualize:
                    print(f"[Act] Saving video for object: {self.obj_goal}")
                    self.save_video_for_object(self.obj_goal)
                self.save_global_scenegraph(tag=f"goal_{self.obj_goal}_{self.target_object_idx}_found")
                self.reset_local_scenegraph()
            else:
                # All objects found
                print(f"[Act] All target objects found! Episode complete.")
                # Save final video for last object
                if self.args.visualize:
                    self.save_video()
                self.save_global_scenegraph(tag="episode_complete")
                return {"action": 0}
            
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            self.goal_map[max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.goal_gps[1]*100/self.resolution))), max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.goal_gps[0]*100/self.resolution)))] = 1
        elif self.found_possible_goal: 
            print(f"[Act] Possible goal found, navigating to it...")
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            self.goal_map[max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.possible_goal_temp_gps[1]*100/self.resolution))), max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.possible_goal_temp_gps[0]*100/self.resolution)))] = 1
        elif self.first_fbe:
            print(f"[Act] First frontier-based exploration, computing FBE goal...")
            self.goal_loc = self.fbe(traversible, cur_start)
            self.not_use_random_goal()
            self.first_fbe = False
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            if self.goal_loc is None:
                print(f"[Act] FBE returned None, using random goal")
                self.random_this_ex += 1
                self.goal_map = self.set_random_goal()
                self.using_random_goal = True
            else:
                print(f"[Act] FBE found frontier at: {self.goal_loc}")
                self.fronter_this_ex += 1
                self.goal_map[self.goal_loc[0], self.goal_loc[1]] = 1
                self.goal_map = self.goal_map[::-1]
        
        # local policy
        print(f"[Act] Computing local policy (short-term goal)...")
        stg_y, stg_x, replan, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        print(f"[Act] Short-term goal: ({stg_y:.2f}, {stg_x:.2f}), Action: {number_action}, Replan: {replan}")
        
        if self.found_possible_goal and number_action == 0:
            print(f"[Act] Possible goal reached, clearing goal flag")
            self.found_possible_goal = False
        
        # reach long-term goal and fbe
        # this is essentially to reselect goal when current goal is unreachable
        print(f"[Act] Checking goal status - found_goal: {self.found_goal}, found_possible_goal: {self.found_possible_goal}, action: {number_action}")
        if (not self.found_goal and not self.found_possible_goal and number_action == 0) or (self.using_random_goal and self.move_since_random > 10): 
            print(f"[Act] Reselecting goal - Goal unreachable or random goal expired")
            if (self.using_random_goal and self.move_since_random > 20):
                print(f"[Act] Random goal expired after {self.move_since_random} steps, blocking previous area")
                goal_x, goal_y = np.where(self.goal_map == 1)
                x_0 = max(goal_x[0] - 8, 0)
                y_0 = max(goal_y[0] - 8, 0)
                x_1 = min(goal_x[0] + 8, self.map_size)
                y_1 = min(goal_y[0] + 8, self.map_size)
                self.fbe_free_map[x_0:x_1, y_0:y_1] = 0
                self.global_fbe_free_map[x_0:x_1, y_0:y_1] = 0
            print(f"[Act] Computing new FBE goal...")
            self.goal_loc = self.fbe(traversible, cur_start)
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            if self.goal_loc is None:
                if hasattr(self.args, 'frontier_teleport') and self.args.frontier_teleport \
                        and self._teleport_to_saved_frontier():
                    print(f"[Frontier Teleport] Teleported during replan! Returning forward action")
                    self.prev_action = 1
                    return {"action": 1}
                print(f"[Act] FBE returned None, using random goal")
                self.random_this_ex += 1
                self.goal_map = self.set_random_goal()
                self.using_random_goal = True
            else:
                print(f"[Act] FBE found new frontier at: {self.goal_loc}")
                self.fronter_this_ex += 1
                self.goal_map[self.goal_loc[0], self.goal_loc[1]] = 1
                self.goal_map = self.goal_map[::-1]
            print(f"[Act] Recomputing local policy with new goal...")
            stg_y, stg_x, replan, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
            print(f"[Act] New short-term goal: ({stg_y:.2f}, {stg_x:.2f}), Action: {number_action}")
        
        # self.loop_time = 0
        # another attempt to replan when stuck
        # This is the issue of being stuck
        # print(f"[Act] Stuck detection - not_move_steps: {self.not_move_steps}, found_goal: {self.found_goal}, action: {number_action}")
        # while (not self.found_goal and number_action == 0) or self.not_move_steps >= 7:
        #     print(f"[Act] Agent stuck! Attempting to unstuck (attempt {self.loop_time + 1}/20)")
        #     if self.not_move_steps >= 7:
        #         print(f"[Act] Stationary for {self.not_move_steps} steps, resetting goal flags")
        #         self.found_goal = False
        #         self.found_possible_goal = False
        #     self.loop_time += 1
        #     self.random_this_ex += 1
        #     if self.loop_time > 20:
        #         print(f"[Act] Failed to unstuck after 20 attempts, giving up")
        #         return {"action": 0}
        #     self.not_move_steps = 0
        #     print(f"[Act] Setting random goal to escape stuck position")
        #     self.goal_map = self.set_random_goal()
        #     self.using_random_goal = True
        #     stg_y, stg_x, replan, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        #     print(f"[Act] Retry {self.loop_time}: New action: {number_action}")
        
        self.loop_time = 0
        # Stuck detection and recovery - IMPROVED VERSION
        print(f"[Act] Stuck detection - not_move_steps: {self.not_move_steps}, found_goal: {self.found_goal}, action: {number_action}")
        
        # Only enter stuck recovery if we're truly stuck (action=0 without goal, OR stationary too long)
        if (not self.found_goal and number_action == 0) or self.not_move_steps >= 4:
            print(f"[Act] Agent stuck! Attempting recovery...")
            
            # If stationary too long, reset goal flags
            if self.not_move_steps >= 4:
                print(f"[Act] Stationary for {self.not_move_steps} steps, resetting goal flags")
                self.found_goal = False
                self.found_possible_goal = False
                self.not_move_steps = 0
                
                # Block the current goal area to prevent returning to it
                goal_x, goal_y = np.where(self.goal_map == 1)
                if len(goal_x) > 0:
                    x_0 = max(goal_x[0] - 10, 0)
                    y_0 = max(goal_y[0] - 10, 0)
                    x_1 = min(goal_x[0] + 10, self.map_size)
                    y_1 = min(goal_y[0] + 10, self.map_size)
                    print(f"[Act] Blocking stuck region: [{x_0}:{x_1}, {y_0}:{y_1}]")
                    self.fbe_free_map[0, 0, x_0:x_1, y_0:y_1] = 0
                    self.global_fbe_free_map[0, 0, x_0:x_1, y_0:y_1] = 0
            
            # === LLM ESCAPE MODE (tried first if enabled) ===
            llm_escaped = False
            if hasattr(self.args, 'llm_escape') and self.args.llm_escape \
                    and self.escape_attempts < self.max_escape_attempts:
                if self.llm_plan_escape(observations):
                    number_action = self.escape_action_queue.pop(0)
                    print(f"[LLM Escape] Starting escape with action: {number_action}")
                    llm_escaped = True
                else:
                    print(f"[LLM Escape] VLM failed, falling back to primitive recovery")

            # === PRIMITIVE RECOVERY (FBE → frontier teleport → random goal) ===
            if not llm_escaped:
                print(f"[Act] Computing new FBE goal to escape...")
                new_goal_loc = self.fbe(traversible, cur_start)
                
                if new_goal_loc is not None and self.toggle_stuck_handling:
                    print(f"[Act] FBE found escape frontier at: {new_goal_loc}")
                    self.goal_map = np.zeros(self.full_map.shape[-2:])
                    self.goal_map[new_goal_loc[0], new_goal_loc[1]] = 1
                    self.goal_map = self.goal_map[::-1]
                    self.using_random_goal = False
                    self.fronter_this_ex += 1
                    self.toggle_stuck_handling = False
                elif hasattr(self.args, 'frontier_teleport') and self.args.frontier_teleport \
                        and self._teleport_to_saved_frontier():
                    print(f"[Frontier Teleport] Teleported! Returning forward action to re-observe")
                    self.prev_action = 1
                    self.toggle_stuck_handling = True  # Allow FBE to select new goal on next step after teleport
                    return {"action": 1}
                else:
                    print(f"[Act] FBE failed, using random goal")
                    self.goal_map = self.set_random_goal()
                    self.using_random_goal = True
                    self.random_this_ex += 1
                
                stg_y, stg_x, replan, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
                print(f"[Act] New plan: STG=({stg_y:.2f}, {stg_x:.2f}), Action={number_action}")
                
                if number_action == 0:
                    self.loop_time += 1
                    if self.loop_time <= 3:
                        if self.loop_time % 2 == 1:
                            number_action = 2
                            print(f"[Act] Forcing turn LEFT to break stuck cycle (attempt {self.loop_time})")
                        else:
                            number_action = 3
                            print(f"[Act] Forcing turn RIGHT to break stuck cycle (attempt {self.loop_time})")
                    elif self.loop_time <= 6:
                        number_action = 1
                        print(f"[Act] Forcing FORWARD movement to break stuck cycle (attempt {self.loop_time})")
                    else:
                        print(f"[Act] Failed to unstuck after {self.loop_time} attempts")
                        self.loop_time = 0
                        number_action = 1
                else:
                    self.loop_time = 0
        else:
            self.loop_time = 0


        if self.args.visualize:
            print(f"[Act] Generating visualization...")
            self.visualize(traversible, observations, number_action)

        observations["pointgoal_with_gps_compass"] = self.get_relative_goal_gps(observations)

        self.last_loc = self.full_pose.clone().detach()
        self.prev_action = number_action
        self.action_history.append(number_action)
        if len(self.action_history) > 20:
            self.action_history = self.action_history[-20:]
        self.navigate_steps += 1
        
        # Print final metrics and statistics
        self.print_metrics()
        _step_total_ms = (time.perf_counter() - _step_t0) * 1000
        print(f"[Act] Final action: {number_action} | Fronts: {self.fronter_this_ex} | Random: {self.random_this_ex} | Move steps: {self.move_steps}")
        print(f"[Act] Objects found: {sum(self.found_objects.values())}/{len(self.target_objects_list)}")
        print(f"[Timing] Step {self.total_steps}: LOCAL_SG={_local_sg_ms:.0f}ms | GLOBAL_SG={_global_sg_ms:.0f}ms | Maps={_maps_ms:.0f}ms | Total={_step_total_ms:.0f}ms")
        print(f"{'='*100}\n")
        
        return {"action": number_action}
    
    def not_use_random_goal(self):
        self.move_since_random = 0
        self.using_random_goal = False
        
    def get_glip_real_label(self, prediction):
        labels = prediction.get_field("labels").tolist()
        new_labels = []
        if self.glip_demo.entities and self.glip_demo.plus:
            for i in labels:
                if i <= len(self.glip_demo.entities):
                    new_labels.append(self.glip_demo.entities[i - self.glip_demo.plus])
                else:
                    new_labels.append('object')
        else:
            new_labels = ['object' for i in labels]
        return new_labels
    
    def _save_frontier_positions(self, frontier_locations, scores):
        """Save top-scoring frontier positions along with current agent 3D state."""
        try:
            sim = self.simulator._env._sim
            agent_state = sim.get_agent_state()
            # Save top 3 scoring frontiers
            top_k = min(3, len(scores))
            top_indices = np.argsort(scores)[-top_k:]
            for idx in top_indices:
                entry = {
                    'map_loc': frontier_locations[idx].copy(),
                    'agent_position': np.array(agent_state.position),
                    'agent_rotation': np.array([agent_state.rotation.x, agent_state.rotation.y,
                                                 agent_state.rotation.z, agent_state.rotation.w]),
                    'step': self.total_steps,
                }
                self.saved_frontier_positions.append(entry)
            # Keep at most 50 saved positions
            if len(self.saved_frontier_positions) > 50:
                self.saved_frontier_positions = self.saved_frontier_positions[-50:]
            print(f"[Frontier Teleport] Saved {top_k} frontier positions "
                  f"(total saved: {len(self.saved_frontier_positions)})")
        except Exception as e:
            print(f"[Frontier Teleport] Error saving frontier positions: {e}")

    def _save_agent_waypoint(self):
        """Save current agent 3D state as a waypoint for potential teleportation."""
        if not (hasattr(self.args, 'frontier_teleport') and self.args.frontier_teleport):
            return
        try:
            sim = self.simulator._env._sim
            agent_state = sim.get_agent_state()
            entry = {
                'map_loc': None,
                'agent_position': np.array(agent_state.position),
                'agent_rotation': np.array([agent_state.rotation.x, agent_state.rotation.y,
                                             agent_state.rotation.z, agent_state.rotation.w]),
                'step': self.total_steps,
            }
            self.saved_frontier_positions.append(entry)
            if len(self.saved_frontier_positions) > 50:
                self.saved_frontier_positions = self.saved_frontier_positions[-50:]
        except Exception as e:
            print(f"[Frontier Teleport] Error saving waypoint: {e}")

    def _teleport_to_saved_frontier(self):
        """Teleport agent to a random saved frontier/waypoint position.
        
        Returns True if teleport succeeded, False otherwise.
        """
        if not self.saved_frontier_positions:
            print(f"[Frontier Teleport] No saved positions to teleport to")
            return False

        if self.frontier_teleport_count >= self.max_frontier_teleports:
            print(f"[Frontier Teleport] Max teleports ({self.max_frontier_teleports}) reached")
            return False

        try:
            sim = self.simulator._env._sim
            current_state = sim.get_agent_state()
            current_pos = np.array(current_state.position)

            # Prefer positions far from current location (at least 1m away)
            candidates = []
            for entry in self.saved_frontier_positions:
                dist = np.linalg.norm(entry['agent_position'] - current_pos)
                if dist > 1.0:
                    candidates.append((entry, dist))

            if not candidates:
                # If no far candidates, use all saved positions
                candidates = [(e, 0.0) for e in self.saved_frontier_positions]

            # Pick randomly, weighted by distance (prefer farther positions)
            distances = np.array([d for _, d in candidates])
            if distances.sum() > 0:
                probs = distances / distances.sum()
            else:
                probs = np.ones(len(candidates)) / len(candidates)
            chosen_idx = np.random.choice(len(candidates), p=probs)
            chosen = candidates[chosen_idx][0]

            target_pos = chosen['agent_position'].tolist()
            target_rot = chosen['agent_rotation'].tolist()

            # Check if position is navigable
            if hasattr(sim, 'is_navigable') and not sim.is_navigable(target_pos):
                print(f"[Frontier Teleport] Target position not navigable, trying nearby point")
                if hasattr(sim, 'pathfinder'):
                    nearby = sim.pathfinder.get_random_navigable_point_near(
                        np.array(target_pos), radius=2.0
                    )
                    if nearby is not None and sim.is_navigable(nearby.tolist()):
                        target_pos = nearby.tolist()
                    else:
                        print(f"[Frontier Teleport] Could not find navigable point nearby")
                        return False
                else:
                    print(f"[Frontier Teleport] Simulator does not support pathfinding, cannot adjust position")
                    return False

            # Teleport
            sim.set_agent_state(target_pos, target_rot)
            self.frontier_teleport_count += 1
            
            # Reinitialize local maps (keep global maps intact)
            self.init_map()
            self.first_fbe = True
            self.found_goal = False
            self.found_possible_goal = False
            self.not_move_steps = 0
            self.loop_time = 0
            self.using_random_goal = False
            self.move_since_random = 0
            
            print(f"[Frontier Teleport] Teleported to saved position from step {chosen['step']} "
                  f"(teleport #{self.frontier_teleport_count}/{self.max_frontier_teleports})")
            print(f"[Frontier Teleport] New position: {target_pos}")
            return True
        except Exception as e:
            print(f"[Frontier Teleport] Teleport failed: {e}")
            return False

    def fbe(self, traversible, start):
        fbe_map = torch.zeros_like(self.full_map[0,0])
        fbe_map[self.fbe_free_map[0,0]>0] = 1 # first free 
        fbe_map[skimage.morphology.binary_dilation(self.full_map[0,0].cpu().numpy(), skimage.morphology.disk(4))] = 3 # then dialte obstacle

        fbe_cp = fbe_map.clone()
        fbe_cpp = fbe_map.clone()
        fbe_cp[fbe_cp==0] = 4 # don't know space is 4
        fbe_cp[fbe_cp<4] = 0 # free and obstacle
        selem = skimage.morphology.disk(1)
        fbe_cpp[skimage.morphology.binary_dilation(fbe_cp.cpu().numpy(), selem)] = 0 # don't know space is 0 dialate unknown space
        
        diff = fbe_map - fbe_cpp # intersection between unknown area and free area 
        frontier_map = diff == 1
        frontier_indices = torch.where(frontier_map)
        frontier_locations = torch.stack([frontier_indices[0], frontier_indices[1]]).T
        num_frontiers = len(frontier_indices[0])
        if num_frontiers == 0:
            return None
        
        # for each frontier, calculate the inverse of distance
        planner = FMMPlanner(traversible, None)
        state = [start[0] + 1, start[1] + 1]
        planner.set_goal(state)
        fmm_dist = planner.fmm_dist[::-1]
        frontier_locations += 1
        frontier_locations = frontier_locations.cpu().numpy()
        distances = fmm_dist[frontier_locations[:,0],frontier_locations[:,1]] / 20
        
        ## use the threshold of 1.6 to filter close frontiers to encourage exploration
        idx_16 = np.where(distances>=1.6)
        distances_16 = distances[idx_16]
        distances_16_inverse = 1 - (np.clip(distances_16,0,11.6)-1.6) / (11.6-1.6)
        frontier_locations_16 = frontier_locations[idx_16]
        self.frontier_locations = frontier_locations
        self.frontier_locations_16 = frontier_locations_16
        if len(distances_16) == 0:
            return None
        num_16_frontiers = len(idx_16[0])  # 175

        scores = self.scenegraph.score(frontier_locations_16, num_16_frontiers)
                
        scores += 2 * distances_16_inverse
        idx_16_max = idx_16[0][np.argmax(scores)]
        goal = frontier_locations[idx_16_max] - 1
        self.scores = scores

        # Save top frontier positions for potential teleportation
        if hasattr(self.args, 'frontier_teleport') and self.args.frontier_teleport:
            self._save_frontier_positions(frontier_locations_16, scores)

        # Clean up GPU tensors
        del fbe_map, fbe_cp, fbe_cpp, diff, frontier_map

        return goal
        
    def get_goal_gps(self, observations, angle, distance):
        if type(angle) is torch.Tensor:
            angle = angle.cpu().numpy()
        agent_gps = observations['gps']
        agent_compass = observations['compass']
        goal_direction = agent_compass - angle/180*np.pi
        goal_gps = np.array([(agent_gps[0]+np.cos(goal_direction)*distance).item(),
         (agent_gps[1]-np.sin(goal_direction)*distance).item()])
        return goal_gps

    def get_relative_goal_gps(self, observations, goal_gps=None):
        if goal_gps is None:
            goal_gps = self.goal_gps
        direction_vector = goal_gps - np.array([observations['gps'][0].item(),observations['gps'][1].item()])
        rho = np.sqrt(direction_vector[0]**2 + direction_vector[1]**2)
        phi_world = np.arctan2(direction_vector[1], direction_vector[0])
        agent_compass = observations['compass']
        phi = phi_world - agent_compass
        return np.array([rho, phi.item()], dtype=np.float32)
   
    def init_global_map(self):
        self.map_size = self.map_size_cm // self.map_resolution
        full_w, full_h = self.map_size, self.map_size
        self.global_full_map = torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.global_room_map = torch.zeros(1,9 ,full_w, full_h).float().to(self.device)
        self.global_visited = self.global_full_map[0,0].cpu().numpy()
        self.global_collision_map = self.global_full_map[0,0].cpu().numpy()

        # self.global_fbe_free_map = copy.deepcopy(self.global_full_map).to(self.device) # 0 is unknown, 1 is free
        self.global_fbe_free_map = self.global_full_map.clone().to(self.device)  # Changed from copy.deepcopy
        self.global_origins = np.zeros((2))
        
    # Added and used global version    
    def init_map(self):
        self.map_size = self.map_size_cm // self.map_resolution
        full_w, full_h = self.map_size, self.map_size
        self.full_map = torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.room_map = torch.zeros(1,9 ,full_w, full_h).float().to(self.device)
        self.visited = self.full_map[0,0].cpu().numpy()
        self.collision_map = self.full_map[0,0].cpu().numpy()

        self.fbe_free_map = self.full_map.clone().to(self.device)  # Changed from copy.deepcopy
        # self.fbe_free_map = copy.deepcopy(self.full_map).to(self.device) # 0 is unknown, 1 is free
        self.full_pose = torch.zeros(3).float().to(self.device)
        self.goal_gps_map = self.full_map[0,0].cpu().numpy()
        self.origins = np.zeros((2))
        
        def init_map_and_pose():
            self.full_map.fill_(0.)
            self.full_pose.fill_(0.)
            self.full_pose[:2] = self.map_size_cm / 100.0 / 2.0  # put the agent in the middle of the map

        init_map_and_pose()

    def _update_pose_once(self, observations):
        """Update pose once and reuse across all map modules"""
        # Convert once
        gps_tensor = torch.from_numpy(observations['gps']).to(self.device)
        compass_tensor = torch.from_numpy(observations['compass'] * 57.29577951308232).to(self.device)
        
        # Update pose
        self.full_pose[0] = self.map_size_cm / 100.0 / 2.0 + gps_tensor[0]
        self.full_pose[1] = self.map_size_cm / 100.0 / 2.0 - gps_tensor[1]
        self.full_pose[2:] = compass_tensor
        
        # Clean up intermediate tensors
        del gps_tensor, compass_tensor
        return self.full_pose

    def update_global_map(self, observations):
        pose = self._update_pose_once(observations)
        self.global_full_map = self.global_sem_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), pose, self.global_full_map)

    # Added and used global version
    def update_map(self, observations):
        pose = self._update_pose_once(observations)
        self.full_map = self.sem_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), pose, self.full_map)
    
    def update_free_map(self, observations):
        pose = self._update_pose_once(observations)
        self.fbe_free_map = self.free_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), pose, self.fbe_free_map)
        self.fbe_free_map[int(self.map_size_cm / 10) - 3:int(self.map_size_cm / 10) + 4, int(self.map_size_cm / 10) - 3:int(self.map_size_cm / 10) + 4] = 1
    
    def update_global_free_map(self, observations):
        pose = self._update_pose_once(observations)
        self.global_fbe_free_map = self.global_free_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), pose, self.global_fbe_free_map)
        self.global_fbe_free_map[int(self.map_size_cm / 10) - 3:int(self.map_size_cm / 10) + 4, int(self.map_size_cm / 10) - 3:int(self.map_size_cm / 10) + 4] = 1 

    # Added and used global version
    def update_room_map(self, observations, room_prediction_result):
        new_room_labels = self.get_glip_real_label(room_prediction_result)
        type_mask = np.zeros((9,self.config.SIMULATOR.DEPTH_SENSOR.HEIGHT, self.config.SIMULATOR.DEPTH_SENSOR.WIDTH))
        bboxs = room_prediction_result.bbox
        score_vec = torch.zeros((9)).to(self.device)
        for i, box in enumerate(bboxs):
            box = box.to(torch.int64)
            idx = rooms.index(new_room_labels[i])
            type_mask[idx,box[1]:box[3],box[0]:box[2]] = 1
            score_vec[idx] = room_prediction_result.get_field("scores")[i]
        self.room_map = self.room_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.room_map, torch.from_numpy(type_mask).to(self.device).type(torch.float32), score_vec)
    
    def update_global_room_map(self, observations, room_prediction_result):
        new_room_labels = self.get_glip_real_label(room_prediction_result)
        type_mask = np.zeros((9,self.config.SIMULATOR.DEPTH_SENSOR.HEIGHT, self.config.SIMULATOR.DEPTH_SENSOR.WIDTH))
        bboxs = room_prediction_result.bbox
        score_vec = torch.zeros((9)).to(self.device)
        for i, box in enumerate(bboxs):
            box = box.to(torch.int64)
            idx = rooms.index(new_room_labels[i])
            type_mask[idx,box[1]:box[3],box[0]:box[2]] = 1
            score_vec[idx] = room_prediction_result.get_field("scores")[i]
        self.global_room_map = self.global_room_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.global_room_map, torch.from_numpy(type_mask).to(self.device).type(torch.float32), score_vec)

    def get_traversible(self, map_pred, pose_pred):
        grid = np.rint(map_pred)
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]
        r, c = start_y, start_x
        start = [int(r*100/self.map_resolution - gy1),
                 int(c*100/self.map_resolution - gx1)]
        start = pu.threshold_poses(start, grid.shape)
        self.visited[gy1:gy2, gx1:gx2][start[0]-2:start[0]+3,
                                       start[1]-2:start[1]+3] = 1
        self.global_visited[gy1:gy2, gx1:gx2][start[0]-2:start[0]+3,
                                       start[1]-2:start[1]+3] = 1
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat
        
        [gx1, gx2, gy1, gy2] = planning_window
        x1, y1, = 0, 0
        x2, y2 = grid.shape

        traversible = skimage.morphology.binary_dilation(
                    grid[y1:y2, x1:x2],
                    self.selem) != True

        if not(traversible[start[0], start[1]]):
            print("Not traversible, step is  ", self.navigate_steps)

        traversible = 1 - traversible
        selem = skimage.morphology.disk(2)
        traversible = skimage.morphology.binary_dilation(
                        traversible, selem)
        traversible[self.collision_map[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 1
        traversible = skimage.morphology.binary_dilation(
                        traversible, selem) != True
        
        traversible[int(start[0]-y1)-1:int(start[0]-y1)+2,
            int(start[1]-x1)-1:int(start[1]-x1)+2] = 1
        traversible = traversible * 1.
        
        traversible[self.visited[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 1
        traversible = add_boundary(traversible)
        return traversible, start, start_o
    
    def _plan(self, traversible, goal_map, agent_pose, start, start_o, goal_found):
        # how to move forward if previous action is move forward
        if self.prev_action == 1:
            x1, y1, t1 = self.last_loc.cpu().numpy()
            x2, y2, t2 = self.full_pose.cpu()
            y1 = self.map_size_cm/100 - y1
            y2 = self.map_size_cm/100 - y2
            t1 = -t1
            t2 = -t2
            buf = 4
            length = 5

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            col_threshold = self.collision_threshold

            if dist < col_threshold: # Collision
                self.former_collide += 1
                for i in range(length):
                    wx = x1 + 0.05 * ((i + buf) * np.cos(np.deg2rad(t1)))
                    wy = y1 + 0.05 * ((i + buf) * np.sin(np.deg2rad(t1)))
                    r, c = wy, wx
                    r = int(round(r * 100 / self.map_resolution))
                    c = int(round(c * 100 / self.map_resolution))
                    [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                    self.collision_map[r,c] = 1
            else:
                self.former_collide = 0

        # This tells what to do next
        stg, replan, stop, = self._get_stg(traversible, start, np.copy(goal_map), goal_found)

        # Deterministic Local Policy
        if stop:
            action = 0
            (stg_y, stg_x) = stg

        else:
            (stg_y, stg_x) = stg
            angle_st_goal = math.degrees(math.atan2(stg_y - start[0],
                                                stg_x - start[1]))
            angle_agent = (start_o)%360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_st_goal- angle_agent)%360.0
            if relative_angle > 180:
                relative_angle -= 360

            if self.former_collide < 10:
                if relative_angle > 16:
                    action = 3 # Right
                elif relative_angle < -16:
                    action = 2 # Left
                else:
                    action = 1
            elif self.prev_action == 1:
                if relative_angle > 0:
                    action = 3 # Right
                else:
                    action = 2 # Left
            else:
                action = 1
            if self.former_collide >= 10 and self.prev_action != 1:
                self.former_collide  = 0
            if stg_y == start[0] and stg_x == start[1]:
                action = 1

        return stg_y, stg_x, replan, action
    
    def _get_stg(self, traversible, start, goal, goal_found):
        """Modified short-term goal planning for exploration"""
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat
        
        goal = add_boundary(goal, value=0)
        original_goal = goal.copy()
        
        centers = []
        if len(np.where(goal !=0)[0]) > 1:
            goal, centers = CH._get_center_goal(goal)
        state = [start[0] + 1, start[1] + 1]
        self.planner = FMMPlanner(traversible, None)
            
        if self.dilation_deg!=0: 
            goal = CH._add_cross_dilation(goal, self.dilation_deg, 3)
            
        if goal_found:
            try:
                goal = CH._block_goal(centers, goal, original_goal, goal_found)
            except:
                goal = self.set_random_goal(goal)

        self.planner.set_multi_goal(goal, state) # time cosuming 

        decrease_stop_cond =0
        if self.dilation_deg >= 6:
            decrease_stop_cond = 0.2 #decrease to 0.2 (7 grids until closest goal)
        
        # need a way to control replan frequency and stop condition for exploration scenario
        stg_y, stg_x, replan, stop = self.planner.get_short_term_goal(state, found_goal = goal_found, decrease_stop_cond=decrease_stop_cond)
        stg_x, stg_y = stg_x - 1, stg_y - 1
        
        return (stg_y, stg_x), replan, stop
    
    def set_random_goal(self):
        obstacle_map = self.full_map.cpu().numpy()[0,0,::-1]
        goal = np.zeros_like(obstacle_map)
        goal_index = np.where((obstacle_map<1))
        np.random.seed(self.total_steps)
        if len(goal_index[0]) != 0:
            i = np.random.choice(len(goal_index[0]), 1)[0]
            h_goal = goal_index[0][i]
            w_goal = goal_index[1][i]
        else:
            h_goal = np.random.choice(goal.shape[0], 1)[0]
            w_goal = np.random.choice(goal.shape[1], 1)[0]
        goal[h_goal, w_goal] = 1
        return goal
    
    def update_metrics(self, metrics):
        self.metrics['distance_to_goal'] = metrics['distance_to_goal']
        self.metrics['spl'] = metrics['spl']
        self.metrics['softspl'] = metrics['softspl']
        if self.simulator._env.episode_over:
            print(f"\n[Episode Over] Episode {self.count_episodes} ended at step {self.total_steps} | "
                  f"Goal: {self.obj_goal} | DTG: {metrics['distance_to_goal']:.2f} | "
                  f"SPL: {metrics['spl']:.3f} | Success: {metrics.get('success', 'N/A')}")
        if self.args.visualize:
            if self.simulator._env.episode_over or self.total_steps == 5000:
                self.save_video()
                self.save_global_scenegraph(tag="episode_end")

    def save_global_scenegraph(self, save_dir=None, tag=None):
        """Save the global scene graph and global maps to disk.
        
        Args:
            save_dir: Directory to save to. Defaults to data/scenegraph_saves/<experiment_name>/
            tag: Optional tag for the filename. Defaults to timestamp.
        """
        import pickle
        if save_dir is None:
            save_dir = os.path.join('data', 'scenegraph_saves', self.experiment_name)
        os.makedirs(save_dir, exist_ok=True)

        if tag is None:
            tag = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Serialize the scene graph
        sg_data = self.global_scenegraph.to_serializable_dict()

        # Collect global maps
        maps_data = {
            'global_full_map': self.global_full_map.cpu().numpy(),
            'global_room_map': self.global_room_map.cpu().numpy(),
            'global_fbe_free_map': self.global_fbe_free_map.cpu().numpy(),
            'global_visited': self.global_visited.copy(),
            'global_collision_map': self.global_collision_map.copy(),
            'global_origins': self.global_origins.copy(),
        }

        episode = getattr(self.simulator._env, 'current_episode', None)
        save_data = {
            'scenegraph': sg_data,
            'maps': maps_data,
            # Episode metadata (useful for DynamicQA manifest alignment)
            'scene_id': getattr(episode, 'scene_id', None),
            'episode_id': getattr(episode, 'episode_id', None),
            # Backwards-compatible key (older pickles used 'episode')
            'episode': getattr(self, 'episode_n', None),
            'episode_n': getattr(self, 'episode_n', None),
            'target_object_idx': self.target_object_idx,
            'found_objects': self.found_objects,
            'total_steps': self.total_steps,
        }

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        filepath = os.path.join(save_dir, f'global_sg_{tag}_{timestamp}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Global scene graph saved to {filepath}')
        return filepath

    def load_global_scenegraph(self, filepath):
        """Load a previously saved global scene graph and global maps.
        
        Args:
            filepath: Path to the .pkl file saved by save_global_scenegraph().
        """
        import pickle
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        # Restore scene graph
        self.global_scenegraph.from_serializable_dict(save_data['scenegraph'])

        # Restore global maps
        maps = save_data['maps']
        self.global_full_map = torch.from_numpy(maps['global_full_map']).float().to(self.device)
        self.global_room_map = torch.from_numpy(maps['global_room_map']).float().to(self.device)
        self.global_fbe_free_map = torch.from_numpy(maps['global_fbe_free_map']).float().to(self.device)
        self.global_visited = maps['global_visited'].copy()
        self.global_collision_map = maps['global_collision_map'].copy()
        self.global_origins = maps['global_origins'].copy()

        # Restore metadata
        self.target_object_idx = save_data.get('target_object_idx', 0)
        self.found_objects = save_data.get('found_objects', {})
        print(f'Global scene graph loaded from {filepath}')

    def visualize(self, traversible, observations, number_action):
        if self.args.visualize:
            # save_map = copy.deepcopy(torch.from_numpy(traversible))
            save_map = torch.from_numpy(traversible).clone()  # Changed from copy.deepcopy
            gray_map = torch.stack((save_map, save_map, save_map))

            # paper_obstacle_map = copy.deepcopy(gray_map)[:,1:-1,1:-1]
            paper_obstacle_map = gray_map[:, 1:-1, 1:-1]  # Direct slicing, no deepcopy
            paper_map = torch.zeros_like(paper_obstacle_map)
            paper_map_trans = paper_map.permute(1,2,0)
            unknown_rgb = colors.to_rgb('#FFFFFF')
            paper_map_trans[:,:,:] = torch.tensor( unknown_rgb)
            free_rgb = colors.to_rgb('#E7E7E7')

            # paper_map_trans[self.fbe_free_map.cpu().numpy()[0,0,::-1]>0.5,:] = torch.tensor( free_rgb).double()
            global_fbe_map_np = self.global_fbe_free_map.cpu().numpy()[0, 0, ::-1]
            paper_map_trans[global_fbe_map_np > 0.5, :] = torch.tensor(free_rgb).double()

            # paper_map_trans[global_fbe_map_np>0.5,:] = torch.tensor( free_rgb).double()
            obstacle_rgb = colors.to_rgb('#A2A2A2')
            paper_map_trans[skimage.morphology.binary_dilation(self.full_map.cpu().numpy()[0,0,::-1]>0.5,skimage.morphology.disk(1)),:] = torch.tensor(obstacle_rgb).double()
            paper_map_trans = paper_map_trans.permute(2,0,1)
            self.visualize_agent_and_goal(paper_map_trans)
            
            # Guard against empty history_pose (can happen after reset_local_scenegraph)
            if len(self.history_pose) > 0:
                agent_coordinate = (int(self.history_pose[-1][0]*100/self.resolution), int((self.map_size_cm/100-self.history_pose[-1][1])*100/self.resolution))
                occupancy_map = crop_around_point((paper_map_trans.permute(1, 2, 0) * 255).numpy().astype(np.uint8), agent_coordinate, (150, 200))
            else:
                # If history_pose is empty, skip visualization for this step
                print(f"[Visualize] Skipping visualization - history_pose is empty (likely after goal reset)")
                return
            visualize_image = np.full((450, 800, 3), 255, dtype=np.uint8)
            visualize_image = add_resized_image(visualize_image, self.rgb_visualization, (10, 60), (320, 240))
            visualize_image = add_resized_image(visualize_image, occupancy_map, (340, 60), (180, 240))
            visualize_image = add_rectangle(visualize_image, (10, 60), (330, 300), (128, 128, 128), thickness=1)
            visualize_image = add_rectangle(visualize_image, (340, 60), (520, 300), (128, 128, 128), thickness=1)
            visualize_image = add_rectangle(visualize_image, (540, 60), (790, 165), (128, 128, 128), thickness=1)
            visualize_image = add_rectangle(visualize_image, (540, 195), (790, 300), (128, 128, 128), thickness=1)
            visualize_image = add_rectangle(visualize_image, (10, 350), (790, 400), (128, 128, 128), thickness=1)
            visualize_image = add_text(visualize_image, "Observation (Goal: {})".format(self.obj_goal), (70, 50), font_scale=0.5, thickness=1)
            visualize_image = add_text(visualize_image, "Occupancy Map", (370, 50), font_scale=0.5, thickness=1)
            visualize_image = add_text(visualize_image, "Scene Graph Nodes", (580, 50), font_scale=0.5, thickness=1)
            visualize_image = add_text(visualize_image, "Scene Graph Edges", (580, 185), font_scale=0.5, thickness=1)
            visualize_image = add_text(visualize_image, "LLM Explanation", (330, 340), font_scale=0.5, thickness=1)
            visualize_image = add_text_list(visualize_image, line_list(self.text_node, 40), (550, 80), font_scale=0.3, thickness=1)
            visualize_image = add_text_list(visualize_image, line_list(self.text_edge, 40), (550, 215), font_scale=0.3, thickness=1)
            visualize_image = add_text_list(visualize_image, line_list(self.explanation, 150), (20, 370), font_scale=0.3, thickness=1)
            visualize_image = visualize_image[:, :, ::-1]
            self.visualize_image_list.append(visualize_image)

            # Clean up GPU tensors
            del save_map, gray_map, paper_obstacle_map, paper_map, paper_map_trans
            torch.cuda.empty_cache()

    def save_video(self):
        if len(self.visualize_image_list) == 0:
            print(f"[Save Video] No visualizations to save")
            return
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_video_dir = os.path.join(self.visualization_dir, 'video')
        save_video_path = f'{save_video_dir}/vid_{timestamp}_ep{self.count_episodes:04d}_idx{self.target_object_idx:02d}.mp4'
        if not os.path.exists(save_video_dir):
            os.makedirs(save_video_dir)
        try:
            height, width, layers = self.visualize_image_list[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(save_video_path, fourcc, 4.0, (width, height))
            for visualize_image in self.visualize_image_list:  
                video.write(visualize_image)
            video.release()
            print(f"[Save Video] Saved {len(self.visualize_image_list)} frames to {save_video_path}")
        except Exception as e:
            print(f"[Save Video] Error: {e}")
        # Free visualization frames from memory
        self.visualize_image_list.clear()
    
    def save_video_for_object(self, obj_name):
        """Save video for a specific object search, then clear frames for the next goal."""
        if len(self.visualize_image_list) == 0:
            print(f"[Save Video] No visualizations to save for object: {obj_name}")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_video_dir = os.path.join(self.visualization_dir, 'video')
        save_video_path = f'{save_video_dir}/vid_{timestamp}_ep{self.count_episodes:04d}_idx{self.target_object_idx:02d}_{obj_name}.mp4'
        
        if not os.path.exists(save_video_dir):
            os.makedirs(save_video_dir)
        
        try:
            height, width, layers = self.visualize_image_list[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(save_video_path, fourcc, 4.0, (width, height))
            for visualize_image in self.visualize_image_list:  
                video.write(visualize_image)
            video.release()
            print(f"[Save Video] Saved video for '{obj_name}' ({len(self.visualize_image_list)} frames) to {save_video_path}")
        except Exception as e:
            print(f"[Save Video] Error saving video for '{obj_name}': {e}")
        # Free visualization frames — next goal starts a fresh video
        self.visualize_image_list.clear()

    def visualize_agent_and_goal(self, map):
        for idx, pose in enumerate(self.history_pose):
            draw_step_num = 30
            alpha = max(0, 1 - (len(self.history_pose) - idx) / draw_step_num)
            agent_size = 1
            if idx == len(self.history_pose) - 1:
                agent_size = 2
            draw_agent(agent=self, map=map, pose=pose, agent_size=agent_size, color_index=0, alpha=alpha)
        draw_goal(agent=self, map=map, goal_size=2, color_index=1)
        return map


def _get_free_and_total_gpu_memory():
    """Get free and total GPU memory in bytes, compatible with older PyTorch."""
    if hasattr(torch.cuda, 'mem_get_info'):
        return torch.cuda.mem_get_info(0)
    # Fallback: parse nvidia-smi output
    import subprocess
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits', '-i', '0'],
        capture_output=True, text=True
    )
    free_mb, total_mb = [int(x.strip()) for x in result.stdout.strip().split(',')]
    return free_mb * 1024**2, total_mb * 1024**2


def _reserve_gpu_memory(reserve_gb=15):
    """Pre-reserve GPU memory so other processes can't claim it.
    PyTorch's caching allocator keeps the memory even after the tensor is freed."""
    if torch.cuda.is_available():
        free_before, total = _get_free_and_total_gpu_memory()
        reserve_bytes = int(reserve_gb * 1024**3)
        reserve_bytes = min(reserve_bytes, int(free_before * 0.95))  # don't exceed 95% of free
        print(f"[GPU Reserve] Reserving {reserve_bytes / 1024**3:.2f} GB of GPU memory "
              f"(free: {free_before / 1024**3:.2f} GB, total: {total / 1024**3:.2f} GB)")
        # intentially leave the tensor allocated for the duration of the program to hold the memory
        dummy = torch.empty(reserve_bytes // 4, dtype=torch.float32, device='cuda:0')
        del dummy
        free_after, _ = _get_free_and_total_gpu_memory()
        print(f"[GPU Reserve] Done. Free before: {free_before / 1024**3:.2f} GB, "
              f"Free after: {free_after / 1024**3:.2f} GB, "
              f"Reserved by PyTorch: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visualize", action='store_true'
    )
    parser.add_argument(
        "--split_l", default=0, type=int
    )
    parser.add_argument(
        "--split_r", default=11, type=int
    )
    parser.add_argument(
        "--llm_escape", action='store_true',
        help="Enable LLM-guided escape when robot is stuck"
    )
    parser.add_argument(
        "--frontier_teleport", action='store_true',
        help="Enable teleporting to saved frontiers when stuck with no new frontiers"
    )
    parser.add_argument(
        "--reserve_gpu_gb", default=15, type=float,
        help="Pre-reserve GPU memory in GB to prevent other processes from claiming it"
    )
    # DynamicQA object injection arguments
    parser.add_argument(
        "--inject_manifest", default=None, type=str,
        help="Path to DynamicQA JSONL manifest for physics-based object injection"
    )
    parser.add_argument(
        "--inject_variant", default=0, type=int,
        help="Which placement variant to use from manifest (0=A, 1=B)"
    )
    parser.add_argument(
        "--inject_record_idx", default=None, type=int,
        help="Inject a specific record index from the manifest (default: auto-match by scene)"
    )
    args = parser.parse_args()
    _reserve_gpu_memory(args.reserve_gpu_gb)
    os.environ["CHALLENGE_CONFIG_FILE"] = "configs/challenge_objectnav2021.local.rgbd.yaml"
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    agent = SG_Nav_Agent(task_config=config, args=args)

    challenge = habitat.Challenge(eval_remote=False, split_l=args.split_l, split_r=args.split_r)

    challenge.submit(agent)


if __name__ == "__main__":
    main()
