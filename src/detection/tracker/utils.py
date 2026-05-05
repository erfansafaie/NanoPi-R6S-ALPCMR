import numpy as np


def tlwh_to_xywh(tlwh):
    x, y, w, h = tlwh
    return np.array([x + w / 2.0, y + h / 2.0, w, h], dtype=np.float32)


def xywh_to_tlwh(xywh):
    cx, cy, w, h = xywh
    return np.array([cx - w / 2.0, cy - h / 2.0, w, h], dtype=np.float32)


def tlbr_to_tlwh(tlbr):
    x1, y1, x2, y2 = tlbr
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)


def tlwh_to_tlbr(tlwh):
    x, y, w, h = tlwh
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def tlbr_to_xywh(tlbr):
    return tlwh_to_xywh(tlbr_to_tlwh(tlbr))


def xywh_to_tlbr(xywh):
    return tlwh_to_tlbr(xywh_to_tlwh(xywh))


def iou_batch(atlbrs, btlbrs):
    """
    IoU between two sets of boxes in TLBR format.
    atlbrs: (N, 4)
    btlbrs: (M, 4)
    return: (N, M)
    """
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

    atlbrs = np.asarray(atlbrs, dtype=np.float32)
    btlbrs = np.asarray(btlbrs, dtype=np.float32)

    area_a = (atlbrs[:, 2] - atlbrs[:, 0]) * (atlbrs[:, 3] - atlbrs[:, 1])
    area_b = (btlbrs[:, 2] - btlbrs[:, 0]) * (btlbrs[:, 3] - btlbrs[:, 1])

    lt = np.maximum(atlbrs[:, None, :2], btlbrs[None, :, :2])
    rb = np.minimum(atlbrs[:, None, 2:], btlbrs[None, :, 2:])

    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / np.clip(union, a_min=1e-6, a_max=None)
    return iou.astype(np.float32)
