import torch
import torch.nn.functional as F
from .slam_classes import MapObjectList, DetectionList, to_tensor
from .utils import compute_overlap_matrix_2set, merge_obj2_into_obj1
from .iou import compute_iou_batch, compute_3d_iou_accuracte_batch



def compute_spatial_similarities(cfg, detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''
    Compute the spatial similarities between the detections and the objects
    
    Args:
        detection_list: a list of M detections
        objects: a list of N objects in the map
    Returns:
        A MxN tensor of spatial similarities
    '''
    det_bboxes = detection_list.get_stacked_values_torch('bbox')
    obj_bboxes = objects.get_stacked_values_torch('bbox')

    if cfg.spatial_sim_type == "iou":
        spatial_sim = compute_iou_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "giou":
        raise NotImplementedError("GIoU batch computation is not implemented")
    elif cfg.spatial_sim_type == "iou_accurate":
        spatial_sim = compute_3d_iou_accuracte_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "giou_accurate":
        raise NotImplementedError("Accurate GIoU batch computation is not implemented")
    elif cfg.spatial_sim_type == "overlap":
        spatial_sim = compute_overlap_matrix_2set(cfg, objects, detection_list)
        spatial_sim = torch.from_numpy(spatial_sim).T
    else:
        raise ValueError(f"Invalid spatial similarity type: {cfg.spatial_sim_type}")
    
    return spatial_sim


def compute_visual_similarities(cfg, detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''
    Compute CLIP visual similarities between new detections and existing objects.
    Uses cosine similarity between CLIP image features (clip_ft).

    Args:
        detection_list: a list of M new detections (each with 'clip_ft')
        objects: a list of N existing objects in the map (each with 'clip_ft')
    Returns:
        A MxN tensor of visual similarities (cosine similarity values)
    '''
    det_feats = detection_list.get_stacked_values_torch('clip_ft')  # (M, D)
    obj_feats = objects.get_stacked_values_torch('clip_ft')         # (N, D)
    
    # Normalize for cosine similarity
    det_feats = F.normalize(det_feats.float(), dim=-1)
    obj_feats = F.normalize(obj_feats.float(), dim=-1)
    
    # (M, D) @ (D, N) -> (M, N)
    visual_sim = det_feats @ obj_feats.T
    
    return visual_sim


def aggregate_similarities(cfg, spatial_sim: torch.Tensor, visual_sim: torch.Tensor) -> torch.Tensor:
    '''
    Combine spatial and visual similarities into an aggregate score.
    DovSG-style: weighted sum of spatial overlap and CLIP visual similarity.
    Spatial is weighted higher to prevent merging visually-similar but
    physically-distinct objects (e.g. two chairs across a room).

    Args:
        spatial_sim: MxN spatial similarity matrix
        visual_sim: MxN visual similarity matrix
    Returns:
        MxN aggregated similarity matrix
    '''
    w_spatial = 0.7
    w_visual = 0.3
    agg_sim = w_spatial * spatial_sim + w_visual * visual_sim
    return agg_sim


def merge_detections_to_objects(
    cfg, 
    detection_list: DetectionList, 
    objects: MapObjectList, 
    agg_sim: torch.Tensor
) -> MapObjectList:
    # Iterate through all detections and merge them into objects
    for i in range(agg_sim.shape[0]):
        # If not matched to any object, add it as a new object
        if agg_sim[i].max() == float('-inf'):
            # detection_list[i]['id'] = len(objects)
            objects.append(detection_list[i])
        # Merge with most similar existing object
        else:
            j = agg_sim[i].argmax()
            matched_det = detection_list[i]
            matched_obj = objects[j]
            merged_obj = merge_obj2_into_obj1(cfg, matched_obj, matched_det, run_dbscan=False)
            objects[j] = merged_obj
            
    return objects