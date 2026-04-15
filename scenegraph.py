import base64
import math
import os
import re
import time
from collections import Counter
from io import BytesIO
from pathlib import Path, PosixPath
import cv2
import numpy as np
import omegaconf
import supervision as sv
import torch
import ollama
from omegaconf import DictConfig
from PIL import Image
from sklearn.cluster import DBSCAN

from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from GroundingDINO.groundingdino.datasets import transforms as T
from transformers import CLIPModel, CLIPProcessor

from utils.utils_scenegraph.mapping import compute_spatial_similarities, compute_visual_similarities, aggregate_similarities, merge_detections_to_objects, dedup_detections, periodic_merge_objects
from utils.utils_scenegraph.slam_classes import MapObjectList
from utils.utils_scenegraph.utils import filter_objects, gobs_to_detection_list, text2value
from utils.utils_scenegraph.grounded_sam_demo import get_grounding_output, load_image, load_model


ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'INFO'
}

ADDITIONAL_CLI_OPTIONS = [
    # '--postgres'
]


class RoomNode():
    def __init__(self, caption):
        self.caption = caption
        self.exploration_level = 0
        self.nodes = set()
        self.group_nodes = []


class GroupNode():
    def __init__(self, caption=''):
        self.caption = caption
        self.exploration_level = 0
        self.corr_score = 0
        self.center = None
        self.center_node = None
        self.nodes = []
        self.edges = set()
    
    def __lt__(self, other):
        return self.corr_score < other.corr_score
    
    def get_graph(self):
        self.center = np.array([node.center for node in self.nodes]).mean(axis=0)
        min_distance = np.inf
        for node in self.nodes:
            distance = np.linalg.norm(np.array(node.center) - np.array(self.center))
            if distance < min_distance:
                min_distance = distance
                self.center_node = node
            self.edges.update(node.edges)
        self.caption = self.graph_to_text(self.nodes, self.edges)

    def graph_to_text(self, nodes, edges, include_snapshots=False):
        nodes_text = ', '.join([node.caption for node in nodes])
        if include_snapshots:
            edges_text = ', '.join([edge.text_with_snapshot() for edge in edges])
        else:
            edges_text = ', '.join([f"{edge.node1.caption} {edge.relation} {edge.node2.caption}" for edge in edges])
        return f"Nodes: {nodes_text}. Edges: {edges_text}."


class ObjectNode():
    _next_id = 0

    @classmethod
    def _gen_id(cls):
        nid = cls._next_id
        cls._next_id += 1
        return nid

    @classmethod
    def reset_id_counter(cls):
        cls._next_id = 0

    def __init__(self, node_id=None):
        self.id = node_id if node_id is not None else ObjectNode._gen_id()
        self.is_new_node = True
        self.is_goal_node = False
        self.caption = None
        self.object = None
        self.reason = None
        self.center = None
        self.room_node = None
        self.exploration_level = 0
        self.distance = 2
        self.score = 0.5
        self.edges = set()

    def __repr__(self):
        return f"ObjectNode(id={self.id}, caption={self.caption!r})"

    def __lt__(self, other):
        return self.score < other.score

    def add_edge(self, edge):
        self.edges.add(edge)

    def remove_edge(self, edge):
        self.edges.discard(edge)
    
    def set_caption(self, new_caption):
        for edge in list(self.edges):
            edge.delete()
        self.is_new_node = True
        self.caption = new_caption
        self.reason = None
        self.distance = 2
        self.score = 0.5
        self.exploration_level = 0
        self.edges.clear()
    
    def set_object(self, object):
        self.object = object
        self.object['node'] = self
    
    def set_center(self, center):
        self.center = center


class Edge():
    SNAPSHOT_SIZE = (224, 224)

    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        node1.add_edge(self)
        node2.add_edge(self)
        self.relation = None
        # Memory snapshot fields (3D-Mem inspired)
        self.snapshot_image = None       # np.ndarray thumbnail (H, W, 3) uint8
        self.snapshot_frame_idx = None   # index into segment2d_results
        self.snapshot_step = None        # navigate_steps when captured
        self.snapshot_bboxes = None      # dict with 'node1': [x1,y1,x2,y2], 'node2': [x1,y1,x2,y2]
        self.snapshot_clip_features = None  # np.ndarray (1, D) CLIP embedding of the snapshot
        # Edge enrichment fields for cross-run matching
        self.distance_3d = None          # float: 3D Euclidean distance between node centroids
        self.height_diff = None          # float: signed height difference (node1.z - node2.z)
        self.n_co_observations = 1       # int: how many frames showed both objects

    def set_relation(self, relation):
        self.relation = relation

    def compute_spatial_metrics(self):
        """Compute 3D distance and height difference between connected nodes.
        
        Uses point cloud centroids when available, falls back to 2D center.
        """
        try:
            c1 = np.asarray(self.node1.object['pcd'].get_center())
            c2 = np.asarray(self.node2.object['pcd'].get_center())
            self.distance_3d = float(np.linalg.norm(c1 - c2))
            self.height_diff = float(c1[2] - c2[2])
        except Exception:
            self.distance_3d = None
            self.height_diff = None

    def set_snapshot(self, image, frame_idx=None, step=None, bboxes=None,
                     clip_features=None):
        """Store a memory snapshot on this edge.
        
        Args:
            image: PIL.Image or np.ndarray — will be resized to SNAPSHOT_SIZE.
            frame_idx: index into segment2d_results for provenance.
            step: navigation step when the snapshot was taken.
            bboxes: dict with 'node1' and 'node2' bounding boxes [x1,y1,x2,y2].
            clip_features: optional pre-computed CLIP embedding (np.ndarray).
        """
        if isinstance(image, Image.Image):
            thumb = image.resize(self.SNAPSHOT_SIZE, Image.LANCZOS)
            self.snapshot_image = np.array(thumb, dtype=np.uint8)
        elif isinstance(image, np.ndarray):
            thumb = Image.fromarray(image).resize(self.SNAPSHOT_SIZE, Image.LANCZOS)
            self.snapshot_image = np.array(thumb, dtype=np.uint8)
        else:
            self.snapshot_image = None
        self.snapshot_frame_idx = frame_idx
        self.snapshot_step = step
        self.snapshot_bboxes = bboxes
        if clip_features is not None:
            self.snapshot_clip_features = np.array(clip_features) if not isinstance(clip_features, np.ndarray) else clip_features
        else:
            self.snapshot_clip_features = None

    @property
    def has_snapshot(self):
        return self.snapshot_image is not None

    def get_snapshot_pil(self):
        """Return the snapshot as a PIL Image, or None."""
        if self.snapshot_image is not None:
            return Image.fromarray(self.snapshot_image)
        return None

    def clear_snapshot(self):
        """Free snapshot memory while keeping relation metadata."""
        self.snapshot_image = None
        self.snapshot_clip_features = None

    def delete(self):
        self.node1.remove_edge(self)
        self.node2.remove_edge(self)

    def text(self):
        text = '({}, {}, {})'.format(self.node1.caption, self.node2.caption, self.relation)
        return text

    def text_with_snapshot(self):
        """Rich text including snapshot metadata for QA prompts."""
        base = '({}, {}, {})'.format(self.node1.caption, self.node2.caption, self.relation)
        if self.has_snapshot:
            meta_parts = []
            if self.snapshot_step is not None:
                meta_parts.append(f"step={self.snapshot_step}")
            if self.snapshot_frame_idx is not None:
                meta_parts.append(f"frame={self.snapshot_frame_idx}")
            if meta_parts:
                base += ' [snapshot: {}]'.format(', '.join(meta_parts))
            else:
                base += ' [snapshot: available]'
        return base


class SceneGraph():
    def __init__(self, map_resolution, map_size_cm, map_size, camera_matrix, is_navigation=True, agent=None, is_global=False, shared_models=None) -> None:
        self.map_resolution = map_resolution
        self.map_size_cm = map_size_cm
        self.map_size = map_size
        full_w, full_h = self.map_size, self.map_size
        self.full_w = full_w
        self.full_h = full_h
        self.visited = torch.zeros(full_w, full_h).float().cpu().numpy()
        self.num_of_goal = torch.zeros(full_w, full_h).int()
        self.camera_matrix = camera_matrix
        self.SAM_ENCODER_VERSION = "vit_h"
        self.sam_variant = 'groundedsam'
        self.device = 'cuda'
        self.classes = ['item']
        self.BG_CLASSES = ["wall", "floor", "ceiling"]
        self.rooms = ['bedroom', 'living room', 'bathroom', 'kitchen', 'dining room', 'office room', 'gym', 'lounge', 'laundry room']
        self.objects = MapObjectList(device=self.device)
        self.objects_post = MapObjectList(device=self.device)
        self.nodes = []
        self.edge_text = ''
        self.edge_list = []
        self.group_nodes = []
        self.init_room_nodes()
        self.reason_visualization = ''
        self.is_navigation = is_navigation
        self.is_global = is_global
        self.llm_name = 'llama3.2-vision'
        self.vlm_name = 'llama3.2-vision'
        self.seg_xyxy = None
        self.seg_caption = None
        
        self.groundingdino_config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        self.groundingdino_checkpoint = 'data/models/groundingdino_swint_ogc.pth'
        self.sam_version = 'vit_h'
        self.sam_checkpoint = 'data/models/sam_vit_h_4b8939.pth'
        self.segment2d_results = []
        self.max_detections_per_object = 10
        
        self.threshold_list = {'bathtub': 1, 'bed': 3, 'cabinet': 2, 'chair': 2, 'chest_of_drawers': 2, 'clothes': 4, 'counter': 2, 'cushion': 3, 'fireplace': 2, 'gym_equipment': 3, 'picture': 4, 'plant': 2, 'seating': 1, 'shower': 1, 'sink': 2, 'sofa': 4, 'stool': 2, 'table': 3, 'toilet': 2, 'towel': 2, 'tv_monitor': 1, 'treadmill': 2, 'fitness equipment': 2,
            'lamp': 2, 'mirror': 2, 'rug': 2, 'curtain': 2, 'shelf': 2, 'desk': 2, 'door': 2, 'window': 2, 'pillow': 2, 'blanket': 2,
            # DynamicQA small/dynamic objects — low thresholds since they're small
            'phone': 1, 'cell phone': 1, 'remote': 1, 'tv remote': 1, 'laptop': 1,
            'book': 1, 'keys': 1, 'mug': 1, 'cup': 1, 'bottle': 1, 'plate': 1, 'bowl': 1}
        self.small_objects = ['bathtub', 'chest_of_drawers', 'cushion', 'plant', 'seating', 'shower', 'toilet', 'tv_monitor',
            'lamp', 'mirror', 'pillow', 'blanket',
            # DynamicQA small/dynamic objects
            'phone', 'cell phone', 'remote', 'tv remote', 'laptop', 'book', 'keys', 'mug', 'cup', 'bottle', 'plate', 'bowl']
        self.store_edge_snapshots = True  # set False to disable memory snapshots on edges
        self.found_goal_times_threshold = 1
        self.N_max = 10
        self.object_vocabulary = self._load_object_vocabulary('tools/object_vocabulary.txt')
        self._vocab_list = [v.strip().lower() for v in self.object_vocabulary]
        self._vocab_set = set(self._vocab_list)
        self._clip_text_cache = {}
        self.node_space = '. '.join(self.object_vocabulary) + '.'
        print(f"[SceneGraph] Loaded {len(self.object_vocabulary)} objects for detection: {self.node_space}")
        self.prompt_edge_proposal = '''
Provide the most possible single spatial relationship for each of the following object pairs. Answer with only one relationship per pair, and separate each answer with a newline character. Do not response superfluous text.
Example 1:
Input:
Object pair(s):
(cabinet, chair)
Output:
next to

Example 2:
Input:
Object pair(s):
(table, lamp)
(bed, nightstand)
Output:
on
next to

Now input is: 
Object pair(s):
        '''
        self.prompt_relation = 'What is the spatial relationship between the {} and the {} in the image? You can only answer a word or phrase that describes a spatial relationship.'
        self.prompt_discriminate_relation = 'In the image, do {} and {} satisfy the relationship of {}? Only answer "yes" or "no".'
        self.prompt_room_predict = 'Which room is the most likely to have the [{}] in: [{}]. Only answer the room.'
        self.prompt_graph_corr_0 = 'What is the probability of A and B appearing together. [A:{}], [B:{}]. Even if you do not have enough information, you have to answer with a value from 0 to 1 anyway. Answer only the value of probability and do not answer any other text.'
        self.prompt_graph_corr_1 = 'What else do you need to know to determine the probability of A and B appearing together? [A:{}], [B:{}]. Please output a short question (output only one sentence with no additional text).'
        self.prompt_graph_corr_2 = 'Here is the objects and relationships near A: [{}] You answer the following question with a short sentence based on this information. Question: {}'
        self.prompt_graph_corr_3 = 'The probability of A and B appearing together is about {}. Based on the dialog: [{}], re-determine the probability of A and B appearing together. A:[{}], B:[{}]. Even if you do not have enough information, you have to answer with a value from 0 to 1 anyway. Answer only the value of probability and do not answer any other text.'

        # --- GPU model loading (or reuse from shared_models) ---
        # GroundingDINO + SAM and CLIP are heavy (~5-8 GB combined).
        # When two SceneGraph instances exist (local + global), the second
        # one can reuse the first's models via shared_models dict to halve
        # GPU memory and eliminate redundant weight loading.
        if shared_models is not None:
            print(f"[SceneGraph] Reusing shared models (GroundingDINO+SAM, CLIP) — no extra GPU memory")
            self.mask_generator = shared_models['mask_generator']
            self.clip_model = shared_models['clip_model']
            self.clip_processor = shared_models['clip_processor']
        else:
            self.mask_generator = self.get_sam_mask_generator(self.sam_variant, self.device)
            clip_model_name = "openai/clip-vit-base-patch32"
            print(f"[SceneGraph] Loading CLIP model: {clip_model_name}")
            self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            self.clip_model.eval()
            print(f"[SceneGraph] CLIP model loaded successfully")
        
        self.set_cfg()
        self.set_agent(agent)
        # Pre-warm Ollama model so first edge call isn't a cold start (~10s load)
        self._warm_ollama()

    def _warm_ollama(self):
        """Pre-load the Ollama model into GPU memory and pin it with keep_alive=-1."""
        try:
            import time
            t0 = time.perf_counter()
            ollama.chat(
                model=self.vlm_name,
                messages=[{'role': 'user', 'content': 'hi'}],
                keep_alive=-1,
            )
            elapsed = time.perf_counter() - t0
            print(f"[SceneGraph] Ollama model '{self.vlm_name}' pre-warmed in {elapsed:.1f}s (pinned with keep_alive=-1)")
        except Exception as e:
            print(f"[SceneGraph] WARNING: Ollama warm-up failed: {e}")

    def _load_object_vocabulary(self, filepath):
        """Load object vocabulary from file for GroundingDINO prompt"""
        objects = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        objects.append(line)
        except FileNotFoundError:
            print(f"[SceneGraph] WARNING: {filepath} not found, using default vocabulary")
            objects = ['bathtub', 'bed', 'cabinet', 'chair', 'drawers', 'clothes', 'counter',
                       'cushion', 'fireplace', 'gym', 'picture', 'plant', 'seating', 'shower',
                       'sink', 'sofa', 'stool', 'table', 'toilet', 'towel', 'tv', 'treadmill',
                       'fitness equipment']
        return objects

    def _normalize_caption_text(self, caption):
        caption = caption.lower().strip().replace('_', ' ')
        caption = re.sub(r'[^a-z0-9 ]+', ' ', caption)
        return ' '.join(caption.split())

    def _extract_caption_candidates(self, caption):
        cleaned = self._normalize_caption_text(caption)
        if not cleaned:
            return []
        if cleaned in self._vocab_set:
            return [cleaned]
        padded = f" {cleaned} "
        candidates = [v for v in self._vocab_list if f" {v} " in padded]
        if candidates:
            seen = set()
            ordered = []
            for item in sorted(candidates, key=len, reverse=True):
                if item not in seen:
                    seen.add(item)
                    ordered.append(item)
            return ordered
        return [cleaned]

    def _get_text_features_cached(self, labels):
        missing = [label for label in labels if label not in self._clip_text_cache]
        if missing:
            inputs = self.clip_processor(text=missing, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                feats = self.clip_model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            for label, feat in zip(missing, feats):
                self._clip_text_cache[label] = feat.cpu()
        return torch.stack([self._clip_text_cache[label] for label in labels], dim=0).to(self.device)

    def _select_caption_for_detection(self, raw_caption, image_feat):
        candidates = self._extract_caption_candidates(raw_caption)
        if not candidates:
            return self._normalize_caption_text(raw_caption)
        if len(candidates) == 1 or image_feat is None:
            return candidates[0]
        image_tensor = torch.as_tensor(image_feat, device=self.device, dtype=torch.float32)
        if image_tensor.ndim == 2 and image_tensor.shape[0] == 1:
            image_tensor = image_tensor.squeeze(0)
        image_tensor = image_tensor / image_tensor.norm(dim=-1, keepdim=True)
        text_feats = self._get_text_features_cached(candidates)
        sims = torch.mv(text_feats, image_tensor)
        best_idx = int(torch.argmax(sims).item())
        return candidates[best_idx]

    def reset(self):
        full_w, full_h = self.map_size, self.map_size
        self.full_w = full_w
        self.full_h = full_h
        self.visited = torch.zeros(full_w, full_h).float().cpu().numpy()
        self.num_of_goal = torch.zeros(full_w, full_h).int()
        self.segment2d_results = []
        self.reason = ''
        self.objects = MapObjectList(device=self.device)
        self.objects_post = MapObjectList(device=self.device)
        self.nodes = []
        self.group_nodes = []
        self.init_room_nodes()
        self.edge_text = ''
        self.edge_list = []
        self.reason_visualization = ''
        ObjectNode.reset_id_counter()

    def set_cfg(self):
        cfg = {'dataset_config': PosixPath('tools/replica.yaml'), 'scene_id': 'room0', 'start': 0, 'end': -1, 'stride': 5, 'image_height': 680, 'image_width': 1200, 'gsa_variant': 'none', 'detection_folder_name': 'gsa_detections_${gsa_variant}', 'det_vis_folder_name': 'gsa_vis_${gsa_variant}', 'color_file_name': 'gsa_classes_${gsa_variant}', 'device': 'cuda', 'use_iou': True, 'spatial_sim_type': 'iou_accurate', 'phys_bias': 0.0, 'match_method': 'sim_sum', 'semantic_threshold': 0.5, 'physical_threshold': 0.5, 'sim_threshold': 1.2, 'use_contain_number': False, 'contain_area_thresh': 0.95, 'contain_mismatch_penalty': 0.5, 'mask_area_threshold': 25, 'mask_conf_threshold': 0.95, 'max_bbox_area_ratio': 0.5, 'skip_bg': True, 'min_points_threshold': 16, 'downsample_voxel_size': 0.01, 'dbscan_remove_noise': True, 'dbscan_eps': 0.1, 'dbscan_min_points': 5, 'obj_min_points': 0, 'obj_min_detections': 2, 'merge_overlap_thresh': 0.7, 'merge_visual_sim_thresh': 0.8, 'merge_text_sim_thresh': 0.8, 'denoise_interval': 20, 'filter_interval': -1, 'merge_interval': 20, 'save_pcd': True, 'save_suffix': 'overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub', 'vis_render': False, 'debug_render': False, 'class_agnostic': True, 'save_objects_all_frames': True, 'render_camera_path': 'replica_room0.json', 'max_num_points': 512}
        cfg = DictConfig(cfg)
        if self.is_navigation:
            # Threshold for aggregate similarity (weighted avg of spatial + visual, range [0,1])
            cfg.sim_threshold = 0.6
            cfg.sim_threshold_spatial = 0.01
        self.cfg = cfg

    def set_agent(self, agent):
        self.agent = agent

    def set_obj_goal(self, obj_goal, obj_goal_sg):
        self.obj_goal = obj_goal
        self.obj_goal_sg = obj_goal_sg
        # Global SG uses a fixed threshold to avoid node spikes on goal switches
        if not self.is_global and self.obj_goal in self.threshold_list:
            self.cfg.obj_min_detections = self.threshold_list[self.obj_goal]

    def set_navigate_steps(self, navigate_steps):
        self.navigate_steps = navigate_steps

    def set_room_map(self, room_map):
        self.room_map = room_map

    def set_fbe_free_map(self, fbe_free_map):
        self.fbe_free_map = fbe_free_map
    
    def set_observations(self, observations):
        self.observations = observations
        self.image_rgb = observations['rgb'].copy()
        self.image_depth = observations['depth'].copy()
        self.pose_matrix = self.get_pose_matrix()

    def set_frontier_map(self, frontier_map):
        self.frontier_map = frontier_map

    def set_full_map(self, full_map):
        self.full_map = full_map

    def set_fbe_free_map(self, fbe_free_map):
        self.fbe_free_map = fbe_free_map

    def set_full_pose(self, full_pose):
        self.full_pose = full_pose

    def get_nodes(self):
        return self.nodes
    
    def get_edges(self):
        edges = set()
        for node in self.nodes:
            edges.update(node.edges)
        edges = list(edges)
        return edges

    def get_edges_with_snapshots(self):
        """Return only edges that have memory snapshots attached."""
        return [e for e in self.get_edges() if e.has_snapshot]

    def get_edge_snapshot_for_qa(self, obj1_caption, obj2_caption):
        """Retrieve the snapshot image and metadata for a specific object pair.
        
        Useful for QA: given two object names, return the visual evidence
        of their relationship.
        
        Returns:
            dict with keys 'relation', 'snapshot_pil', 'step', 'frame_idx',
            'bboxes', 'clip_features', or None if no matching edge found.
        """
        for edge in self.get_edges():
            captions = {edge.node1.caption, edge.node2.caption}
            if obj1_caption in captions and obj2_caption in captions:
                result = {
                    'relation': edge.relation,
                    'snapshot_pil': edge.get_snapshot_pil(),
                    'step': edge.snapshot_step,
                    'frame_idx': edge.snapshot_frame_idx,
                    'bboxes': edge.snapshot_bboxes,
                    'clip_features': edge.snapshot_clip_features,
                }
                return result
        return None

    def get_seg_xyxy(self):
        return self.seg_xyxy

    def get_seg_caption(self):
        return self.seg_caption

    def init_room_nodes(self):
        room_nodes = []
        for caption in self.rooms:
            room_node = RoomNode(caption)
            room_nodes.append(room_node)
        self.room_nodes = room_nodes

    def get_sam_mask_generator(self, variant:str, device) -> SamAutomaticMaskGenerator:
        if variant == "sam":
            sam = sam_model_registry[self.SAM_ENCODER_VERSION](checkpoint=self.sam_checkpoint)
            sam.to(device)
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=12,
                points_per_batch=144,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                crop_n_layers=0,
                min_mask_region_area=100,
            )
            return mask_generator
        elif variant == "fastsam":
            raise NotImplementedError
            # from ultralytics import YOLO
            # from FastSAM.tools import *
            # FASTSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/FastSAM-x.pt")
            # model = YOLO(args.model_path)
            # return model
        elif variant == "groundedsam":
            model = load_model(self.groundingdino_config_file, self.groundingdino_checkpoint, None, device=device)
            predictor = SamPredictor(sam_model_registry[self.sam_version](checkpoint=self.sam_checkpoint).to(device))
            return model, predictor
        else:
            raise NotImplementedError
    
    def get_sam_segmentation_dense(
        self, variant:str, model, image: np.ndarray
    ) -> tuple:
        '''
        The SAM based on automatic mask generation, without bbox prompting
        
        Args:
            model: The mask generator or the YOLO model
            image: )H, W, 3), in RGB color space, in range [0, 255]
            
        Returns:
            mask: (N, H, W)
            xyxy: (N, 4)
            conf: (N,)
        '''
        if variant == "sam":
            results = model.generate(image)  # type(results) == list
            mask = []
            xyxy = []
            conf = []
            for r in results:  # type(r) == dict
                mask.append(r["segmentation"])  # type(r["segmentation"]) == np.ndarray, r["segmentation"] == [480, 640]
                r_xyxy = r["bbox"].copy()  # type(r["bbox"]) == list, [x, y, h, w]
                # Convert from xyhw format to xyxy format
                r_xyxy[2] += r_xyxy[0]
                r_xyxy[3] += r_xyxy[1]
                xyxy.append(r_xyxy)
                conf.append(r["predicted_iou"])  # type(r["predicted_iou"]) == float
            mask = np.array(mask)
            xyxy = np.array(xyxy)
            conf = np.array(conf)
            return mask, xyxy, conf
        elif variant == "fastsam":
            # The arguments are directly copied from the GSA repo
            results = model(
                image,
                imgsz=1024,
                device="cuda",
                retina_masks=True,
                iou=0.9,
                conf=0.4,
                max_det=100,
            )
            raise NotImplementedError
        elif variant == "groundedsam":
            groundingdino = model[0]
            sam_predictor = model[1]
            transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            image_resized, _ = transform(Image.fromarray(image), None)  # 3, h, w
            boxes_filt, caption = get_grounding_output(groundingdino, image_resized, caption=self.node_space, box_threshold=0.3, text_threshold=0.25, with_logits=False, device=self.device)
            if len(caption) == 0:
                return None, None, None, None
            sam_predictor.set_image(image)

            # size = image_pil.size
            H, W = image.shape[0], image.shape[1]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

            mask, conf, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(self.device),
                multimask_output = False,
            )
            mask, xyxy, conf = mask.squeeze(1).cpu().numpy(), boxes_filt.squeeze(1).numpy(), conf.squeeze(1).cpu().numpy()

            # Release SAM's cached image tensors (they're large).
            # Do NOT call torch.cuda.empty_cache() here — it's expensive
            # (~20-50ms) and the allocator can reuse the memory on its own.
            sam_predictor.reset_image()
        
            return mask, xyxy, conf, caption
        else:
            raise NotImplementedError

    def compute_clip_features(self, image, detections, classes):
        """Compute CLIP image and text features for each detection.
        Uses HuggingFace CLIPModel + CLIPProcessor.
        Batched: all image crops processed in one GPU forward pass,
        text features cached per unique class label.
        Returns: image_crops, image_feats (N, D), text_feats (N, D) as numpy arrays.
        """
        image = Image.fromarray(image)
        padding = 20
        image_width, image_height = image.size
        
        # --- Collect all crops ---
        image_crops = []
        for idx in range(len(detections.xyxy)):
            x_min, y_min, x_max, y_max = detections.xyxy[idx]
            left_padding = min(padding, x_min)
            top_padding = min(padding, y_min)
            right_padding = min(padding, image_width - x_max)
            bottom_padding = min(padding, image_height - y_max)
            x_min -= left_padding
            y_min -= top_padding
            x_max += right_padding
            y_max += bottom_padding
            image_crops.append(image.crop((x_min, y_min, x_max, y_max)))
        
        if len(image_crops) == 0:
            return image_crops, np.empty((0, 512)), np.empty((0, 512))
        
        # --- Batched image features (single GPU forward pass) ---
        image_inputs = self.clip_processor(images=image_crops, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_feats = self.clip_model.get_image_features(**image_inputs)
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        image_feats = image_feats.cpu().numpy()
        del image_inputs
        
        # --- Cached text features (one forward pass per unique label) ---
        if not hasattr(self, '_clip_text_cache'):
            self._clip_text_cache = {}
        
        unique_cids = set(detections.class_id)
        for cid in unique_cids:
            label = classes[cid]
            if label not in self._clip_text_cache:
                text_inputs = self.clip_processor(text=[label], return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    tf = self.clip_model.get_text_features(**text_inputs)
                tf = tf / tf.norm(dim=-1, keepdim=True)
                self._clip_text_cache[label] = tf.squeeze(0).cpu()
                del text_inputs
        
        text_feats = np.array([self._clip_text_cache[classes[cid]].numpy() for cid in detections.class_id])

        return image_crops, image_feats, text_feats

    def process_cfg(self, cfg: DictConfig):
        cfg.dataset_root = Path(cfg.dataset_root)
        cfg.dataset_config = Path(cfg.dataset_config)
        
        if cfg.dataset_config.name != "multiscan.yaml":
            # For datasets whose depth and RGB have the same resolution
            # Set the desired image heights and width from the dataset config
            dataset_cfg = omegaconf.OmegaConf.load(cfg.dataset_config)
            if cfg.image_height is None:
                cfg.image_height = dataset_cfg.camera_params.image_height
            if cfg.image_width is None:
                cfg.image_width = dataset_cfg.camera_params.image_width
            print(f"Setting image height and width to {cfg.image_height} x {cfg.image_width}")
        else:
            # For dataset whose depth and RGB have different resolutions
            assert cfg.image_height is not None and cfg.image_width is not None, \
                "For multiscan dataset, image height and width must be specified"

        return cfg

    def crop_image_and_mask(self, image: Image, mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: int = 0):
        """ Crop the image and mask with some padding. I made a single function that crops both the image and the mask at the same time because I was getting shape mismatches when I cropped them separately.This way I can check that they are the same shape."""
        
        image = np.array(image)
        # Verify initial dimensions
        if image.shape[:2] != mask.shape:
            print("Initial shape mismatch: Image shape {} != Mask shape {}".format(image.shape, mask.shape))
            return None, None

        # Define the cropping coordinates
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        # round the coordinates to integers
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

        # Crop the image and the mask
        image_crop = image[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]

        # Verify cropped dimensions
        if image_crop.shape[:2] != mask_crop.shape:
            print("Cropped shape mismatch: Image crop shape {} != Mask crop shape {}".format(image_crop.shape, mask_crop.shape))
            return None, None
        
        # convert the image back to a pil image
        image_crop = Image.fromarray(image_crop)

        return image_crop, mask_crop
    
    def get_pose_matrix(self):
        x = self.map_size_cm / 100.0 / 2.0 + self.observations['gps'][0]
        y = self.map_size_cm / 100.0 / 2.0 - self.observations['gps'][1]
        t = (self.observations['compass'] - np.pi / 2)[0] # input degrees and meters
        pose_matrix = np.array([
            [np.cos(t), -np.sin(t), 0, x],
            [np.sin(t), np.cos(t), 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return pose_matrix

    def segment2d(self):
        if self.sam_variant == 'sam' or self.sam_variant == 'groundedsam':
            mask, xyxy, conf, caption = self.get_sam_segmentation_dense(self.sam_variant, self.mask_generator, self.image_rgb)
            self.seg_xyxy = xyxy
            self.seg_caption = caption
            if caption is None:
                print(f"[Segment2D] No detections (caption=None), skipping frame")
                return
            print(f"[Segment2D] Detected {len(mask)} segments: {caption}")
            # Build per-detection class_id from GroundingDINO captions.
            # Previously all detections got class_id=0 → class_name='item';
            # now each gets its real label so dedup can gate on category.
            # Normalize captions: lowercase, strip punctuation, map to vocabulary when possible.
            norm_captions = []
            for c in caption:
                candidates = self._extract_caption_candidates(c)
                norm_captions.append(candidates[0] if candidates else self._normalize_caption_text(c))
            unique_classes = sorted(set(norm_captions))
            class_id_arr = np.array([unique_classes.index(c) for c in norm_captions], dtype=int)
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=class_id_arr,
                mask=mask,
            )
            with torch.no_grad():
                image_crops, image_feats, text_feats = self.compute_clip_features(self.image_rgb, detections, unique_classes)
            image_appear_efficiency = [''] * len(image_crops)
            self.segment2d_results.append({
                "xyxy": detections.xyxy,
                "confidence": detections.confidence,
                "class_id": detections.class_id,
                "mask": detections.mask,
                "classes": unique_classes,
                "image_feats": image_feats,
                "text_feats": text_feats,
                "image_appear_efficiency": image_appear_efficiency,
                "image_rgb": self.image_rgb,
                "caption": caption,
            })

    def mapping3d(self):
        depth_array = self.image_depth
        depth_array = depth_array[..., 0]
        gobs = self.segment2d_results[-1]
        cam_K = self.camera_matrix
            
        idx = len(self.segment2d_results) - 1

        fg_detection_list, bg_detection_list = gobs_to_detection_list(
            cfg = self.cfg,
            image = self.image_rgb,
            depth_array = depth_array,
            cam_K = cam_K,
            idx = idx,
            gobs = gobs,
            trans_pose = self.pose_matrix,
            class_names = gobs['classes'],
            BG_CLASSES = self.BG_CLASSES,
            is_navigation = self.is_navigation,
            navigate_step = getattr(self, 'navigate_steps', None),
            # color_path = color_path,
        )
        
        if len(fg_detection_list) == 0:
            print(f"[Mapping3D] No foreground detections after 3D mapping")
            return
            
        if len(self.objects) == 0:
            # Dedup detections within this frame before adding
            fg_detection_list = dedup_detections(
                self.cfg, fg_detection_list,
                centroid_thresh=0.5, visual_thresh=0.5)
            # Add all detections to the map
            for i in range(len(fg_detection_list)):
                self.objects.append(fg_detection_list[i])
            print(f"[Mapping3D] First frame: added {len(fg_detection_list)} objects (no prior objects to match)")

            # Skip the similarity computation 
            self.objects_post = filter_objects(self.cfg, self.objects)
            return
                
        spatial_sim = compute_spatial_similarities(self.cfg, fg_detection_list, self.objects)
        visual_sim = compute_visual_similarities(self.cfg, fg_detection_list, self.objects)
        agg_sim = aggregate_similarities(self.cfg, spatial_sim, visual_sim)

        # Log per-detection similarity scores before thresholding
        num_new = 0
        num_merged = 0
        for i in range(agg_sim.shape[0]):
            best_j = agg_sim[i].argmax().item()
            best_spatial = spatial_sim[i, best_j].item()
            best_visual = visual_sim[i, best_j].item()
            best_agg = agg_sim[i, best_j].item()
            det_caption = fg_detection_list[i].get('class_name', '?')
            obj_caption = self.objects[best_j].get('class_name', '?') if best_j < len(self.objects) else '?'
            if best_agg < self.cfg.sim_threshold:
                num_new += 1
                print(f"[Mapping3D]   NEW object '{det_caption}' | best match '{obj_caption}' "
                      f"spatial={best_spatial:.3f} visual={best_visual:.3f} agg={best_agg:.3f} < threshold={self.cfg.sim_threshold}")
            else:
                num_merged += 1
                print(f"[Mapping3D]   MERGED '{det_caption}' -> '{obj_caption}' "
                      f"spatial={best_spatial:.3f} visual={best_visual:.3f} agg={best_agg:.3f} >= threshold={self.cfg.sim_threshold}")
        print(f"[Mapping3D] {len(fg_detection_list)} detections: {num_new} new, {num_merged} merged into existing (total objects: {len(self.objects)})")
        
        # Threshold combined sim. Set to negative infinity if below threshold
        agg_sim[agg_sim < self.cfg.sim_threshold] = float('-inf')
        
        self.objects = merge_detections_to_objects(self.cfg, fg_detection_list, self.objects, agg_sim)
        self.objects_post = filter_objects(self.cfg, self.objects)

        # Clean up intermediate tensors
        del spatial_sim, visual_sim, agg_sim, fg_detection_list, bg_detection_list
            
    def get_caption(self):
        if self.sam_variant == 'groundedsam':
            for idx, object in enumerate(self.objects_post):
                caption_list = []
                for idx_det in range(len(object["image_idx"])):
                    image_idx = object["image_idx"][idx_det]
                    mask_idx = object["mask_idx"][idx_det]
                    entry = self.segment2d_results[image_idx]
                    caption = entry['caption'][mask_idx]
                    image_feats = entry.get('image_feats')
                    image_feat = image_feats[mask_idx] if image_feats is not None else None
                    caption = self._select_caption_for_detection(caption, image_feat)
                    caption_list.append(caption)
                caption = self.find_modes(caption_list)[0]
                object['captions'] = [caption]

    def update_node(self):
        # update nodes
        for i, node in enumerate(self.nodes):
            caption_ori = node.caption
            caption_new = node.object['captions'][0]
            if caption_ori != caption_new:
                print(f"[UpdateNode] Renamed node '{caption_ori}' -> '{caption_new}'")
                node.set_caption(caption_new)
        # add new nodes
        new_objects = list(filter(lambda object: 'node' not in object, self.objects_post))
        if new_objects:
            new_captions = [obj['captions'][0] for obj in new_objects]
            print(f"[UpdateNode] Adding {len(new_objects)} new nodes: {new_captions}")
        for new_object in new_objects:
            new_node = ObjectNode()
            caption = new_object['captions'][0]
            new_node.set_caption(caption)
            new_node.set_object(new_object)
            self.nodes.append(new_node)
        existing_captions = [n.caption for n in self.nodes]
        print(f"[UpdateNode] Total nodes now: {len(self.nodes)} -> {existing_captions}")
        # get node.center and node.room
        for node in self.nodes:
            points = np.asarray(node.object['pcd'].points)
            center = points.mean(axis=0)
            x = int(center[0] * 100 / self.map_resolution)
            y = int(center[1] * 100 / self.map_resolution)
            y = self.map_size - 1 - y
            node.set_center([x, y])
            if 0 <= x < self.map_size and 0 <= y < self.map_size and hasattr(self, 'room_map'):
                if sum(self.room_map[0, :, y, x]!=0).item() == 0:
                    room_label = 0
                else:
                    room_label = torch.where(self.room_map[0, :, y, x]!=0)[0][0].item()
            else:
                room_label = 0
            if node.room_node is not self.room_nodes[room_label]:
                if node.room_node is not None:
                    node.room_node.nodes.discard(node)
                node.room_node = self.room_nodes[room_label]
                node.room_node.nodes.add(node)
            if node.caption in self.obj_goal_sg:
                node.is_goal_node = True

    def update_edge(self):
        old_nodes = []
        new_nodes = []
        for i, node in enumerate(self.nodes):
            if node.is_new_node:
                new_nodes.append(node)
                node.is_new_node = False
            else:
                old_nodes.append(node)
        if len(new_nodes) == 0:
            return
        # create the edge between new_node and old_node
        new_edges = []
        for i, new_node in enumerate(new_nodes):
            for j, old_node in enumerate(old_nodes):
                new_edge = Edge(new_node, old_node)
                new_edges.append(new_edge)
        # create the edge between new_node
        for i, new_node1 in enumerate(new_nodes):
            for j, new_node2 in enumerate(new_nodes[i + 1:]):
                new_edge = Edge(new_node1, new_node2)
                new_edges.append(new_edge)
        # get all new_edges
        new_edges = set()
        for i, node in enumerate(self.nodes):
            node_new_edges = set(filter(lambda edge: edge.relation is None, node.edges))
            new_edges = new_edges | node_new_edges
        new_edges = list(new_edges)
        for new_edge in new_edges:
            image, frame_idx, bboxes = self.get_joint_image(
                new_edge.node1, new_edge.node2, return_metadata=True)
            if image is not None:
                prompt = self.prompt_relation.format(new_edge.node1.caption, new_edge.node2.caption)
                response = self.get_vlm_response(prompt=prompt, image=image)
                response = response.replace('.', '').lower()
                new_edge.set_relation(response)
                new_edge.compute_spatial_metrics()
                # Capture memory snapshot
                if self.store_edge_snapshots:
                    step = getattr(self, 'navigate_steps', None)
                    clip_feat = self._compute_snapshot_clip(image)
                    new_edge.set_snapshot(
                        image, frame_idx=frame_idx, step=step,
                        bboxes=bboxes, clip_features=clip_feat)
        new_edges = set()
        for i, node in enumerate(self.nodes):
            node_new_edges = set(filter(lambda edge: edge.relation is None, node.edges))
            new_edges = new_edges | node_new_edges
        new_edges = list(new_edges)
        # get all relation proposals
        if len(new_edges) > 0:
            node_pairs = []
            for new_edge in new_edges:
                node_pairs.append(new_edge.node1.caption)
                node_pairs.append(new_edge.node2.caption)
            prompt = self.prompt_edge_proposal + '\n({}, {})' * len(new_edges)
            prompt = prompt.format(*node_pairs)
            relations = self.get_llm_response(prompt=prompt)
            relations = relations.split('\n')
            if len(relations) == len(new_edges):
                for i, relation in enumerate(relations):
                    new_edges[i].set_relation(relation)
                    new_edges[i].compute_spatial_metrics()
            # discriminate all relation proposals
            self.free_map = self.fbe_free_map.cpu().numpy()[0,0,::-1].copy() > 0.5
            for i, new_edge in enumerate(new_edges):
                if new_edge.relation == None or not self.discriminate_relation(new_edge):
                    new_edge.delete()

    def update_group(self):
        for room_node in self.room_nodes:
            if len(room_node.nodes) > 0:
                room_node.group_nodes = []
                object_nodes = list(room_node.nodes)
                centers = [object_node.center for object_node in object_nodes]
                centers = np.array(centers)
                dbscan = DBSCAN(eps=10, min_samples=1)  
                clusters = dbscan.fit_predict(centers)  
                for i in range(clusters.max() + 1):
                    group_node = GroupNode()
                    indices = np.where(clusters == i)[0]
                    for index in indices:
                        group_node.nodes.append(object_nodes[index])
                    group_node.get_graph()
                    room_node.group_nodes.append(group_node)

    def insert_goal(self, goal=None):
        if goal is None:
            goal = self.obj_goal_sg
        self.update_group()
        room_node_text = ''
        for room_node in self.room_nodes:
            if len(room_node.group_nodes) > 0:
                room_node_text = room_node_text + room_node.caption + ','
        # room_node_text[-2] = '.'
        if room_node_text == '':
            return None
        prompt = self.prompt_room_predict.format(goal, room_node_text)
        response = self.get_llm_response(prompt=prompt)
        response = response.lower()
        predict_room_node = None
        for room_node in self.room_nodes:
            if len(room_node.group_nodes) > 0 and room_node.caption.lower() in response:
                predict_room_node = room_node
        if predict_room_node is None:
            return None
        for group_node in predict_room_node.group_nodes:
            corr_score = self.graph_corr(goal, group_node)
            group_node.corr_score = corr_score
        sorted_group_nodes = sorted(predict_room_node.group_nodes)
        self.mid_term_goal = sorted_group_nodes[-1].center
        return self.mid_term_goal
    
    def _compact_old_segment2d_results(self, keep_recent=5):
        """Strip heavy data (masks, image_rgb) from old segment2d entries to free memory.
        Keeps 'caption' intact since get_caption() references it by absolute index."""
        if len(self.segment2d_results) <= keep_recent:
            return
        for i in range(len(self.segment2d_results) - keep_recent):
            entry = self.segment2d_results[i]
            if entry is None:
                continue
            if 'mask' in entry and entry['mask'] is not None:
                entry['mask'] = None
            if 'image_rgb' in entry and entry['image_rgb'] is not None:
                entry['image_rgb'] = None
            if 'xyxy' in entry and entry['xyxy'] is not None:
                entry['xyxy'] = None
            if 'confidence' in entry and entry['confidence'] is not None:
                entry['confidence'] = None
            if 'image_feats' in entry and entry['image_feats'] is not None:
                entry['image_feats'] = None
            if 'text_feats' in entry and entry['text_feats'] is not None:
                entry['text_feats'] = None

    def update_scenegraph(self):
        print(f'Navigate Step: {self.navigate_steps}', end='\r')
        _t0 = time.perf_counter()
        self.segment2d()
        _seg_ms = (time.perf_counter() - _t0) * 1000
        if len(self.segment2d_results) > 0:
            _t = time.perf_counter()
            self.mapping3d()
            _map3d_ms = (time.perf_counter() - _t) * 1000
            _t = time.perf_counter()
            self.get_caption()
            _caption_ms = (time.perf_counter() - _t) * 1000
            _t = time.perf_counter()
            self.update_node()
            _node_ms = (time.perf_counter() - _t) * 1000
            _t = time.perf_counter()
            self.update_edge()
            _edge_ms = (time.perf_counter() - _t) * 1000
            _total = (time.perf_counter() - _t0) * 1000
            print(f"[SG Timing] seg2d={_seg_ms:.0f}ms map3d={_map3d_ms:.0f}ms caption={_caption_ms:.0f}ms node={_node_ms:.0f}ms edge={_edge_ms:.0f}ms total={_total:.0f}ms")

        # Periodic dedup pass every merge_interval steps (default 20)
        merge_interval = getattr(self.cfg, 'merge_interval', 20)
        if (self.navigate_steps > 0 and
                self.navigate_steps % merge_interval == 0 and
                len(self.objects) > 1):
            prev_count = len(self.objects)
            self.objects = periodic_merge_objects(
                self.cfg, self.objects,
                centroid_thresh=0.5, visual_thresh=0.6)
            if len(self.objects) < prev_count:
                self.objects_post = filter_objects(self.cfg, self.objects)
    
        # Strip heavy data from old segment2d entries
        self._compact_old_segment2d_results()
        # Only flush GPU cache periodically (every 10 steps) rather than every step.
        # torch.cuda.empty_cache() forces the CUDA allocator to return memory to the OS
        # which is expensive (~20-50ms) and usually unnecessary since PyTorch reuses
        # freed blocks on its own.
        if self.navigate_steps % 10 == 0:
            torch.cuda.empty_cache()

    def get_llm_response(self, prompt):
        response = ollama.chat(
            model=self.llm_name,
            messages=[{
                'role': 'user',
                'content': prompt,
            }],
            keep_alive=-1,
        )
        return response.message.content
    
    def get_vlm_response(self, prompt, image):
        buffered = BytesIO()
        image.save(buffered, format='PNG')
        image_bytes = base64.b64encode(buffered.getvalue())
        image_str = str(image_bytes, 'utf-8')
        response = ollama.chat(
            model=self.vlm_name,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_str]
            }],
            keep_alive=-1,
        )
        return response.message.content
        
    def find_modes(self, lst):  
        if len(lst) == 0:
            return ['object']
        else:
            counts = Counter(lst)  
            max_count = max(counts.values())  
            modes = [item for item, count in counts.items() if count == max_count]  
            return modes  
        
    def get_joint_image(self, node1, node2, return_metadata=False):
        image_idx1 = node1.object["image_idx"]
        image_idx2 = node2.object["image_idx"]
        image_idx = set(image_idx1) & set(image_idx2)
        if len(image_idx) == 0:
            return (None, None, None) if return_metadata else None
        conf_max = -np.inf
        idx_max = None
        # get joint images of the two nodes
        for idx in image_idx:
            # Skip compacted entries whose image_rgb has been freed
            if idx >= len(self.segment2d_results) or self.segment2d_results[idx] is None:
                continue
            if self.segment2d_results[idx].get("image_rgb") is None:
                continue
            conf1 = node1.object["conf"][image_idx1.index(idx)]
            conf2 = node2.object["conf"][image_idx2.index(idx)]
            conf = conf1 + conf2
            if conf > conf_max:
                conf_max = conf
                idx_max = idx
        if idx_max is None:
            return (None, None, None) if return_metadata else None
        image = self.segment2d_results[idx_max]["image_rgb"]
        image = Image.fromarray(image)
        if not return_metadata:
            return image
        # Collect per-node bounding boxes from this frame
        bboxes = {}
        seg = self.segment2d_results[idx_max]
        for tag, node, img_idxs in [('node1', node1, image_idx1), ('node2', node2, image_idx2)]:
            if idx_max in img_idxs:
                det_pos = img_idxs.index(idx_max)
                mask_pos = node.object["mask_idx"][det_pos]
                if seg.get("xyxy") is not None and mask_pos < len(seg["xyxy"]):
                    bboxes[tag] = seg["xyxy"][mask_pos].tolist()
        return image, idx_max, bboxes

    def _compute_snapshot_clip(self, image):
        """Compute CLIP image features for an edge snapshot image."""
        if not hasattr(self, 'clip_model') or self.clip_model is None:
            return None
        try:
            pil_img = image if isinstance(image, Image.Image) else Image.fromarray(image)
            inputs = self.clip_processor(images=pil_img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feat = self.clip_model.get_image_features(**inputs)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.cpu().numpy()
        except Exception:
            return None

    def score(self, frontier_locations_16, num_16_frontiers):
        scores = np.zeros((num_16_frontiers))
        for i, loc in enumerate(frontier_locations_16):
            sub_room_map = self.agent.room_map[0,:,max(0,loc[0]-12):min(self.agent.map_size-1,loc[0]+13), max(0,loc[1]-12):min(self.agent.map_size-1,loc[1]+13)].cpu().numpy() # sub_room_map.shape = [9, 25, 25], select the room map around the frontier
            whether_near_room = np.max(np.max(sub_room_map, 1),1)
            score_1 = np.clip(1-(1-self.agent.prob_array_room)-(1-whether_near_room), 0, 10)
            score_2 = 1- np.clip(self.agent.prob_array_room+(1-whether_near_room), -10,1)
            scores[i] = np.sum(score_1) - np.sum(score_2)
        for i in range(len(self.agent.obj_locations)):
            num_obj = len(self.agent.obj_locations[i])
            if num_obj <= 0:
                continue
            frontier_location_mtx = np.tile(frontier_locations_16, (num_obj,1,1))
            obj_location_mtx = np.array(self.agent.obj_locations[i])[:,1:]
            obj_confidence_mtx = np.tile(np.array(self.agent.obj_locations[i])[:,0],(num_16_frontiers,1)).transpose(1,0)
            obj_location_mtx = np.tile(obj_location_mtx, (num_16_frontiers,1,1)).transpose(1,0,2)
            dist_frontier_obj = np.square(frontier_location_mtx - obj_location_mtx)
            dist_frontier_obj = np.sqrt(np.sum(dist_frontier_obj, axis=2)) / 20
            near_frontier_obj = dist_frontier_obj < 1.6
            obj_confidence_mtx[near_frontier_obj==False] = 0
            obj_confidence_max = np.max(obj_confidence_mtx, axis=0)
            score_1 = np.clip(1-(1-self.agent.prob_array_obj[i])-(1-obj_confidence_max), 0, 10)
            score_2 = 1- np.clip(self.agent.prob_array_obj[i]+(1-obj_confidence_max), -10,1)
            scores += score_1 - score_2

        predict_goal_xy = self.insert_goal()
        if predict_goal_xy is not None:
            predict_goal_xy = np.array(predict_goal_xy).reshape(1, 2)
            distance = np.linalg.norm(predict_goal_xy - frontier_locations_16, axis=1)
            score = np.tile(1, (num_16_frontiers))
            score[distance > 32] = 0
            score = score / distance
            scores += score
        return scores

    def discriminate_relation(self, edge):
        image = self.get_joint_image(edge.node1, edge.node2)
        if image is not None:
            response = self.get_vlm_response(self.prompt_discriminate_relation.format(edge.node1.caption, edge.node2.caption, edge.relation), image)
            if 'yes' in response.lower():
                return True
            else:
                return False
        else:
            if edge.node1.room_node != edge.node2.room_node:
                return False
            x1, y1 = edge.node1.center
            x2, y2 = edge.node2.center
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance > self.map_size // 40:
                return False
            alpha = math.atan2(y2 - y1, x2 - x1)  
            sin_2alpha = 2 * math.sin(alpha) * math.cos(alpha)
            if not -0.05 < sin_2alpha < 0.05:
                return False
            n = 3
            for i in range(1, n):
                x = int(x1 + (x2 - x1) * i / n)
                y = int(y1 + (y2 - y1) * i / n)
                if not self.free_map[y, x]:
                    return False
            return True
        
    def perception(self):
        if not self.agent.found_goal:
            self.agent.detect_objects(self.observations)
            # Run room detection every 2 steps, but only if the agent has moved
            # (room layout doesn't change when stationary — saves one GLIP call)
            agent_moved = self.agent.not_move_steps == 0
            if self.agent.total_steps % 2 == 0 and (agent_moved or self.agent.total_steps <= 22):
                room_detection_result = self.agent.glip_demo.inference(self.observations["rgb"][:,:,[2,1,0]], self.agent.rooms_captions)
                self.agent.update_room_map(self.observations, room_detection_result)

    def graph_corr(self, goal, graph):
        prompt = self.prompt_graph_corr_0.format(graph.center_node.caption, goal)
        response_0 = self.get_llm_response(prompt=prompt)
        prompt = self.prompt_graph_corr_1.format(graph.center_node.caption, goal)
        response_1 = self.get_llm_response(prompt=prompt)
        prompt = self.prompt_graph_corr_2.format(graph.caption, response_1)
        response_2 = self.get_llm_response(prompt=prompt)
        prompt = self.prompt_graph_corr_3.format(response_0, response_1 + response_2, graph.center_node.caption, goal)
        response_3 = self.get_llm_response(prompt=prompt)
        corr_score = text2value(response_3)
        return corr_score

    def _serialize_object_dict(self, obj):
        """Convert a single detected_object dict to a serializable form."""
        s = {}
        for k, v in obj.items():
            if k == 'pcd':
                s['pcd_points'] = np.asarray(v.points)
                s['pcd_colors'] = np.asarray(v.colors)
            elif k == 'bbox':
                s['bbox_points'] = np.asarray(v.get_box_points())
                s['bbox_color'] = list(v.color)
            elif k == 'node':
                continue  # skip circular back-reference
            elif k == 'mask':
                # store masks as compressed booleans
                s['mask'] = [m.astype(bool) if isinstance(m, np.ndarray) else m for m in v]
            elif isinstance(v, np.ndarray):
                s[k] = v.tolist()
            elif isinstance(v, (torch.Tensor,)):
                s[k] = v.cpu().numpy().tolist()
            else:
                s[k] = v
        return s

    @staticmethod
    def _deserialize_object_dict(s):
        """Reconstruct a detected_object dict from serialized form."""
        import open3d as o3d
        obj = {}
        for k, v in s.items():
            if k in ('pcd_points', 'pcd_colors', 'bbox_points', 'bbox_color'):
                continue
            elif k == 'mask':
                obj['mask'] = [np.array(m, dtype=bool) if isinstance(m, list) else m for m in v]
            elif k == 'inst_color':
                obj[k] = np.array(v) if isinstance(v, list) else v
            elif k in ('clip_ft', 'text_ft', 'clip_ft_variance', 'clip_ft_mean_sq'):
                obj[k] = torch.tensor(v) if isinstance(v, list) else v
            elif k == 'height_range' and isinstance(v, list):
                obj[k] = tuple(v)
            else:
                obj[k] = v
        # Reconstruct open3d objects
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(s['pcd_points']))
        if 'pcd_colors' in s and len(s['pcd_colors']) > 0:
            pcd.colors = o3d.utility.Vector3dVector(np.array(s['pcd_colors']))
        obj['pcd'] = pcd
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(np.array(s['bbox_points'])))
        bbox.color = s.get('bbox_color', [0, 1, 0])
        obj['bbox'] = bbox
        return obj

    def to_serializable_dict(self):
        """Serialize the scene graph to a plain dict suitable for pickle/json."""
        # Build node index map
        node_to_idx = {id(node): i for i, node in enumerate(self.nodes)}

        # Serialize nodes
        s_nodes = []
        for node in self.nodes:
            s_node = {
                'id': node.id,
                'caption': node.caption,
                'center': list(node.center) if node.center is not None else None,
                'score': node.score,
                'distance': node.distance,
                'exploration_level': node.exploration_level,
                'is_goal_node': node.is_goal_node,
                'is_new_node': node.is_new_node,
                'reason': node.reason,
                'room_idx': self.room_nodes.index(node.room_node) if node.room_node is not None else None,
            }
            if node.object is not None:
                s_node['object'] = self._serialize_object_dict(node.object)
            else:
                s_node['object'] = None
            s_nodes.append(s_node)

        # Serialize edges (deduplicated)
        seen_edges = set()
        s_edges = []
        for node in self.nodes:
            for edge in node.edges:
                eid = id(edge)
                if eid in seen_edges:
                    continue
                seen_edges.add(eid)
                idx1 = node_to_idx.get(id(edge.node1))
                idx2 = node_to_idx.get(id(edge.node2))
                if idx1 is not None and idx2 is not None:
                    s_edge = {
                        'node1_idx': idx1,
                        'node2_idx': idx2,
                        'relation': edge.relation,
                        'distance_3d': edge.distance_3d,
                        'height_diff': edge.height_diff,
                        'n_co_observations': getattr(edge, 'n_co_observations', 1),
                    }
                    # Serialize memory snapshot
                    if edge.has_snapshot:
                        buf = BytesIO()
                        Image.fromarray(edge.snapshot_image).save(buf, format='JPEG', quality=85)
                        s_edge['snapshot_b64'] = base64.b64encode(buf.getvalue()).decode('ascii')
                        s_edge['snapshot_frame_idx'] = edge.snapshot_frame_idx
                        s_edge['snapshot_step'] = edge.snapshot_step
                        s_edge['snapshot_bboxes'] = edge.snapshot_bboxes
                        if edge.snapshot_clip_features is not None:
                            s_edge['snapshot_clip_features'] = edge.snapshot_clip_features.tolist()
                    s_edges.append(s_edge)

        # Serialize group nodes (per room)
        s_groups = []
        for rn in self.room_nodes:
            for gn in rn.group_nodes:
                member_idxs = [node_to_idx[id(n)] for n in gn.nodes if id(n) in node_to_idx]
                s_groups.append({
                    'caption': gn.caption,
                    'exploration_level': gn.exploration_level,
                    'corr_score': gn.corr_score,
                    'center': list(gn.center) if gn.center is not None else None,
                    'room_idx': self.room_nodes.index(rn),
                    'node_idxs': member_idxs,
                })

        # Serialize room nodes
        s_rooms = []
        for rn in self.room_nodes:
            s_rooms.append({
                'caption': rn.caption,
                'exploration_level': rn.exploration_level,
            })

        # Serialize objects_post (with node index for reliable back-reference restoration)
        s_objects_post = []
        for obj in self.objects_post:
            s_obj = self._serialize_object_dict(obj)
            node_ref = obj.get('node')
            if node_ref is not None and id(node_ref) in node_to_idx:
                s_obj['_node_idx'] = node_to_idx[id(node_ref)]
            else:
                s_obj['_node_idx'] = None
            s_objects_post.append(s_obj)

        return {
            'nodes': s_nodes,
            'edges': s_edges,
            'room_nodes': s_rooms,
            'group_nodes': s_groups,
            'objects_post': s_objects_post,
            'visited': self.visited.copy(),
            'num_of_goal': self.num_of_goal.cpu().numpy(),
            'edge_text': self.edge_text,
        }

    def from_serializable_dict(self, data):
        """Restore scene graph state from a serialized dict. 
        Assumes self is already initialized (models, cfg, etc.)."""
        # Restore room nodes
        self.init_room_nodes()
        for i, sr in enumerate(data['room_nodes']):
            self.room_nodes[i].exploration_level = sr['exploration_level']

        # Restore nodes
        self.nodes = []
        max_id = -1
        for s_node in data['nodes']:
            restored_id = s_node.get('id')
            node = ObjectNode(node_id=restored_id)
            if restored_id is not None and restored_id > max_id:
                max_id = restored_id
            node.caption = s_node['caption']
            node.center = s_node['center']
            node.score = s_node['score']
            node.distance = s_node['distance']
            node.exploration_level = s_node['exploration_level']
            node.is_goal_node = s_node['is_goal_node']
            node.is_new_node = s_node['is_new_node']
            node.reason = s_node['reason']
            if s_node['room_idx'] is not None:
                node.room_node = self.room_nodes[s_node['room_idx']]
                self.room_nodes[s_node['room_idx']].nodes.add(node)
            if s_node['object'] is not None:
                obj = self._deserialize_object_dict(s_node['object'])
                obj['node'] = node  # restore back-reference
                node.object = obj
            self.nodes.append(node)

        # Ensure new nodes created after deserialization get unique IDs
        ObjectNode._next_id = max(ObjectNode._next_id, max_id + 1)

        # Restore edges
        self.edge_list = []
        for s_edge in data['edges']:
            n1 = self.nodes[s_edge['node1_idx']]
            n2 = self.nodes[s_edge['node2_idx']]
            edge = Edge(n1, n2)  # auto-adds to both nodes
            edge.set_relation(s_edge['relation'])
            # Restore edge enrichment fields
            edge.distance_3d = s_edge.get('distance_3d')
            edge.height_diff = s_edge.get('height_diff')
            edge.n_co_observations = s_edge.get('n_co_observations', 1)
            # Restore memory snapshot
            if 'snapshot_b64' in s_edge:
                img_bytes = base64.b64decode(s_edge['snapshot_b64'])
                pil_img = Image.open(BytesIO(img_bytes))
                clip_feat = None
                if 'snapshot_clip_features' in s_edge:
                    clip_feat = np.array(s_edge['snapshot_clip_features'])
                edge.set_snapshot(
                    pil_img,
                    frame_idx=s_edge.get('snapshot_frame_idx'),
                    step=s_edge.get('snapshot_step'),
                    bboxes=s_edge.get('snapshot_bboxes'),
                    clip_features=clip_feat,
                )
            self.edge_list.append(edge)

        # Restore objects_post
        self.objects_post = MapObjectList(device=self.device)
        for s_obj in data['objects_post']:
            obj = self._deserialize_object_dict(s_obj)
            # Link back to node using stored index (reliable) or fallback to class/center match
            node_idx = s_obj.get('_node_idx')
            if node_idx is not None and 0 <= node_idx < len(self.nodes):
                obj['node'] = self.nodes[node_idx]
            else:
                # Fallback for data serialized before _node_idx was added
                for node in self.nodes:
                    if node.object is not None and node.object.get('class_name') == obj.get('class_name'):
                        if node.center == obj.get('center', None):
                            obj['node'] = node
                            break
            self.objects_post.append(obj)

        # Restore visited and num_of_goal
        self.visited = data['visited'].copy()
        self.num_of_goal = torch.from_numpy(data['num_of_goal']).int()
        self.edge_text = data.get('edge_text', '')
        self.reason_visualization = ''
        self.group_nodes = []
        self.segment2d_results = []
        self.objects = MapObjectList(device=self.device)
