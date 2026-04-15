import torch
import torch.nn.functional as F
import numpy as np
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


def dedup_detections(cfg, detection_list: DetectionList,
                     centroid_thresh: float = 0.5,
                     visual_thresh: float = 0.5) -> DetectionList:
    """Deduplicate detections within a single frame before adding to the map.

    Merges pairs that share the same category AND have centroids within
    centroid_thresh metres AND CLIP cosine similarity >= visual_thresh.
    Greedy: iterates once, merging into the first survivor.

    Returns a new (smaller or equal) DetectionList.
    """
    n = len(detection_list)
    if n <= 1:
        return detection_list

    # Pre-compute centroids and CLIP features
    centroids = []
    clip_feats = []
    class_names = []
    for det in detection_list:
        pts = np.asarray(det['pcd'].points)
        centroids.append(pts.mean(axis=0) if len(pts) > 0 else np.zeros(3))
        clip_feats.append(to_tensor(det['clip_ft']))
        cn = det.get('class_name', '')
        if isinstance(cn, list):
            print(f"[Dedup] Warning: detection has multiple class names {cn}, using the first one for deduplication")
            cn = cn[0] if cn else ''
        class_names.append(cn.lower().strip())

    centroids = np.stack(centroids)  # (N, 3)
    clip_feats = torch.stack(clip_feats)  # (N, D)
    clip_feats = F.normalize(clip_feats.float(), dim=-1)

    merged_into = list(range(n))  # union-find parent

    def find(x):
        while merged_into[x] != x:
            merged_into[x] = merged_into[merged_into[x]]
            x = merged_into[x]
        return x

    # Compare all pairs — cheap for typical frame sizes (< 30 detections)
    for i in range(n):
        ri = find(i)
        for j in range(i + 1, n):
            rj = find(j)
            if ri == rj:
                continue
            # Same category check
            if class_names[i] != class_names[j]:
                continue
            # Centroid distance
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist > centroid_thresh:
                continue
            # CLIP visual similarity
            sim = (clip_feats[i] @ clip_feats[j]).item()
            if sim < visual_thresh:
                continue
            # Merge j into i (the earlier detection survives)
            merged_into[rj] = ri
            print(f"[Dedup]   Merging detection {j} '{class_names[j]}' into {ri} "
                  f"(dist={dist:.3f}m, clip={sim:.3f})")

    # Build deduplicated list by merging point clouds
    survivors = {}  # root_idx -> detection object
    for i in range(n):
        root = find(i)
        if root not in survivors:
            survivors[root] = detection_list[i]
        else:
            survivors[root] = merge_obj2_into_obj1(
                cfg, survivors[root], detection_list[i], run_dbscan=False)

    result = DetectionList(list(survivors.values()))
    n_merged = n - len(result)
    if n_merged > 0:
        print(f"[Dedup] Removed {n_merged} duplicate detections "
              f"({n} -> {len(result)})")
    return result


def periodic_merge_objects(cfg, objects: MapObjectList,
                           centroid_thresh: float = 0.5,
                           visual_thresh: float = 0.6) -> MapObjectList:
    """Post-hoc deduplication pass over the full object list.

    Finds pairs of existing objects that share the same caption AND have
    centroids within centroid_thresh metres AND CLIP cosine >= visual_thresh,
    then merges them.  Runs in O(N^2) which is fine for typical scene sizes
    (< 500 objects).

    Returns a new (smaller or equal) MapObjectList.
    """
    print(f"[PeriodicMerge] Running periodic merge on {len(objects)} objects...")
    n = len(objects)
    if n <= 1:
        return objects

    # Pre-compute per-object data
    centroids = []
    clip_feats = []
    captions = []
    for obj in objects:
        pts = np.asarray(obj['pcd'].points)
        centroids.append(pts.mean(axis=0) if len(pts) > 0 else np.zeros(3))
        clip_feats.append(to_tensor(obj['clip_ft']))
        cn = obj.get('class_name', '')
        if isinstance(cn, list):
            print(f"[Dedup] Warning: detection has multiple class names {cn}, using the first one for deduplication")
            cn = cn[0] if cn else ''
        captions.append(cn.lower().strip())

    centroids = np.stack(centroids)
    clip_feats = torch.stack(clip_feats)
    clip_feats = F.normalize(clip_feats.float(), dim=-1)

    merged_into = list(range(n))

    def find(x):
        while merged_into[x] != x:
            merged_into[x] = merged_into[merged_into[x]]
            x = merged_into[x]
        return x

    n_merges = 0
    for i in range(n):
        ri = find(i)
        for j in range(i + 1, n):
            rj = find(j)
            if ri == rj:
                continue
            if captions[i] != captions[j]:
                continue
            dist = np.linalg.norm(centroids[ri] - centroids[rj])
            if dist > centroid_thresh:
                continue
            sim = (clip_feats[ri] @ clip_feats[rj]).item()
            if sim < visual_thresh:
                continue
            merged_into[rj] = ri
            n_merges += 1
            print(f"[PeriodicMerge] Merging obj {j} '{captions[j]}' into {ri} "
                  f"(dist={dist:.3f}m, clip={sim:.3f})")

    if n_merges == 0:
        return objects

    # Rebuild object list, merging point clouds
    survivors = {}
    order = []
    for i in range(n):
        root = find(i)
        if root not in survivors:
            survivors[root] = objects[i]
            order.append(root)
        else:
            survivors[root] = merge_obj2_into_obj1(
                cfg, survivors[root], objects[i], run_dbscan=True)

    result = MapObjectList([survivors[r] for r in order])
    print(f"[PeriodicMerge] Merged {n_merges} duplicate pairs "
          f"({n} -> {len(result)} objects)")
    return result