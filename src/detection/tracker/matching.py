import numpy as np


def tlbr_iou(a, b):
    """
    a: (N,4) tlbr
    b: (M,4) tlbr
    returns IoU matrix (N,M)
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    a_x1 = a[:, 0][:, None]
    a_y1 = a[:, 1][:, None]
    a_x2 = a[:, 2][:, None]
    a_y2 = a[:, 3][:, None]

    b_x1 = b[:, 0][None, :]
    b_y1 = b[:, 1][None, :]
    b_x2 = b[:, 2][None, :]
    b_y2 = b[:, 3][None, :]

    inter_x1 = np.maximum(a_x1, b_x1)
    inter_y1 = np.maximum(a_y1, b_y1)
    inter_x2 = np.minimum(a_x2, b_x2)
    inter_y2 = np.minimum(a_y2, b_y2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(0.0, a_x2 - a_x1) * np.maximum(0.0, a_y2 - a_y1)
    area_b = np.maximum(0.0, b_x2 - b_x1) * np.maximum(0.0, b_y2 - b_y1)

    union = area_a + area_b - inter_area
    union = np.maximum(union, 1e-6)

    return (inter_area / union).astype(np.float32)


def iou_distance(tracks, detections, class_aware=False):
    """
    Cost = 1 - IoU
    Lower is better.
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    track_boxes = np.asarray([t.tlbr for t in tracks], dtype=np.float32)
    det_boxes = np.asarray([d.tlbr for d in detections], dtype=np.float32)

    ious = tlbr_iou(track_boxes, det_boxes)
    cost = 1.0 - ious

    if class_aware:
        track_cls = np.asarray([t.cls for t in tracks], dtype=np.int32)
        det_cls = np.asarray([d.cls for d in detections], dtype=np.int32)
        mismatch = track_cls[:, None] != det_cls[None, :]
        cost[mismatch] = 1e6

    return cost.astype(np.float32)


def greedy_match(cost_matrix, thresh):
    """
    Greedy minimum-cost bipartite matching.

    Args:
        cost_matrix: shape (num_tracks, num_dets)
        thresh: max acceptable cost

    Returns:
        matches: list of (row, col)
        unmatched_rows: list[int]
        unmatched_cols: list[int]
    """
    num_rows, num_cols = cost_matrix.shape

    if num_rows == 0 or num_cols == 0:
        return [], list(range(num_rows)), list(range(num_cols))

    flat_order = np.argsort(cost_matrix, axis=None)

    row_used = np.zeros(num_rows, dtype=bool)
    col_used = np.zeros(num_cols, dtype=bool)

    matches = []

    for flat_idx in flat_order:
        r = flat_idx // num_cols
        c = flat_idx % num_cols

        if row_used[r] or col_used[c]:
            continue

        if cost_matrix[r, c] > thresh:
            break

        row_used[r] = True
        col_used[c] = True
        matches.append((r, c))

    unmatched_rows = np.where(~row_used)[0].tolist()
    unmatched_cols = np.where(~col_used)[0].tolist()

    return matches, unmatched_rows, unmatched_cols
