from __future__ import annotations

import numpy as np
import scipy
import lap


def bbox_ioa(box1: np.ndarray, box2: np.ndarray, iou: bool = False, eps: float = 1e-7) -> np.ndarray:
    """Calculate the intersection over box2 area given box1 and box2.

    Args:
        box1 (np.ndarray): A numpy array of shape (N, 4) representing N bounding boxes in x1y1x2y2 format.
        box2 (np.ndarray): A numpy array of shape (M, 4) representing M bounding boxes in x1y1x2y2 format.
        iou (bool, optional): Calculate the standard IoU if True else return inter_area/box2_area.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (np.ndarray): A numpy array of shape (N, M) representing the intersection over box2 area.
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)

    # Box2 area
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # Intersection over box2 area
    return inter_area / (area + eps)


def linear_assignment(cost_matrix: np.ndarray, thresh: float):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.arange(cost_matrix.shape[0], dtype=np.int32),
            np.arange(cost_matrix.shape[1], dtype=np.int32),
        )

    cost_matrix = np.asarray(cost_matrix, dtype=np.float32)
    matches = []
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)

    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])

    matches = np.asarray(matches, dtype=np.int32) if matches else np.empty((0, 2), dtype=np.int32)
    unmatched_a = np.where(x < 0)[0].astype(np.int32)
    unmatched_b = np.where(y < 0)[0].astype(np.int32)
    return matches, unmatched_a, unmatched_b


def iou_distance(atracks, btracks) -> np.ndarray:
    na, nb = len(atracks), len(btracks)
    if na == 0 or nb == 0:
        return np.zeros((na, nb), dtype=np.float32)

    if isinstance(atracks[0], np.ndarray):
        atlbrs = np.ascontiguousarray(atracks, dtype=np.float32)
    else:
        atlbrs = np.ascontiguousarray([track.xyxy for track in atracks], dtype=np.float32)

    if isinstance(btracks[0], np.ndarray):
        btlbrs = np.ascontiguousarray(btracks, dtype=np.float32)
    else:
        btlbrs = np.ascontiguousarray([track.xyxy for track in btracks], dtype=np.float32)

    ious = bbox_ioa(atlbrs, btlbrs, iou=True).astype(np.float32, copy=False)
    return 1.0 - ious


def _xyxy_arrays(tracks_or_boxes):
    if len(tracks_or_boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    if isinstance(tracks_or_boxes[0], np.ndarray):
        return np.ascontiguousarray(tracks_or_boxes, dtype=np.float32)

    return np.ascontiguousarray([t.xyxy for t in tracks_or_boxes], dtype=np.float32)




def embedding_distance(tracks, detections, metric: str = "cosine") -> np.ndarray:
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix

    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)

    cost_matrix = np.maximum(
        0.0,
        scipy.spatial.distance.cdist(track_features, det_features, metric),
    ).astype(np.float32, copy=False)

    return cost_matrix


def fuse_score(cost_matrix: np.ndarray, detections) -> np.ndarray:
    if cost_matrix.size == 0:
        return cost_matrix

    iou_sim = 1.0 - cost_matrix
    det_scores = np.asarray([det.score for det in detections], dtype=np.float32)
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)

    fuse_sim = iou_sim * det_scores
    return 1.0 - fuse_sim
