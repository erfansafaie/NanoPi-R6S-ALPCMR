
import time
from dataclasses import dataclass
import cv2
import numpy as np
from rknnlite.api import RKNNLite


@dataclass(slots=True)
class LetterBoxInfo:
    org_shape: tuple
    new_shape: tuple
    w_ratio: float
    h_ratio: float
    dh: float
    dw: float
    pad_color: tuple


class InferenceDetRKNN:
    def __init__(
        self,
        model_path: str,
        img_size: tuple[int, int],
        model_branch=3,
        obj_thresh=0.3,
        nms_thresh=0.6,
        pre_nms_topk=100,
        max_det=0,
        keep_multi_class=False,
        reg_max=16,
        npu_core=0,
        use_dfl=False,
    ):
        self.model_path = model_path
        self.rknn_model = None
        self.core_mask_list = (RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2)
        self.npu_core = self.core_mask_list[npu_core]
        self.img_size = img_size

        self.letter_box_info = None

        self.model_branch = model_branch

        self.nms_thresh = nms_thresh
        self.obj_thresh = obj_thresh
        self.max_det = max_det
        self.keep_multi_class = keep_multi_class
        self.pre_nms_topk = pre_nms_topk

        if max_det == 0:
            self.keep_multi_class = False

        self.use_dfl = use_dfl
        self.reg_max = reg_max

        self._grid_cache = {}

        # Pre-compute DFL projection vector once
        if self.use_dfl:
            self._dfl_project = np.arange(self.reg_max, dtype=np.float32)

        self.setup_rknn_model()

    def letter_box_preprc(self, img: np.ndarray, pad_color: tuple = (0, 0, 0)) -> np.ndarray:
        h, w = img.shape[:2]
        scale = min(self.img_size[0] / h, self.img_size[1] / w)

        new_h = int(h * scale)
        new_w = int(w * scale)

        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = (self.img_size[0] - new_h) // 2
        pad_w = (self.img_size[1] - new_w) // 2

        letterboxed_img = cv2.copyMakeBorder(
            resized_img,
            pad_h, self.img_size[0] - new_h - pad_h,
            pad_w, self.img_size[1] - new_w - pad_w,
            cv2.BORDER_CONSTANT,
            value=pad_color,
        )

        self.letter_box_info = LetterBoxInfo(
            (h, w), self.img_size, scale, scale, pad_h, pad_w, pad_color
        )
        # Expand dims without copy
        return letterboxed_img[np.newaxis]

    def setup_rknn_model(self):
        self.rknn_model = RKNNLite()
        ret = self.rknn_model.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")
        ret = self.rknn_model.init_runtime(core_mask=self.npu_core)
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime: {ret}")

    def run(self, img: np.ndarray):
        return self.rknn_model.inference([img])

    def release(self):
        self.rknn_model.release()
        self.rknn_model = None

    def filter_boxes(self, boxes, box_cls_probs):
        """Fused filter: scores = max class prob (conf is always 1.0 here)."""
        scores = box_cls_probs.max(axis=-1)
        mask = scores >= self.obj_thresh
        return boxes[mask], box_cls_probs[mask].argmax(axis=-1), scores[mask]

    def nms_boxes(self, boxes, scores):
        """Standard greedy NMS."""
        if boxes.shape[0] == 0:
            return np.empty((0,), dtype=np.int32)

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break

            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])

            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            ovr = inter / (areas[i] + areas[rest] - inter + 1e-6)
            order = rest[ovr <= self.nms_thresh]

        return np.array(keep, dtype=np.int32)

    def box_process(self, position):
        grid_h, grid_w = position.shape[2], position.shape[3]
        grid, stride = self.get_grid_and_stride(grid_h, grid_w)

        if self.use_dfl:
            position = self.dfl(position)

        # Vectorized decode: center +/- dist * stride
        center = (grid + 0.5) * stride
        xy1 = center - position[:, 0:2] * stride
        xy2 = center + position[:, 2:4] * stride

        # Clip in-place
        xyxy = np.concatenate((xy1, xy2), axis=1)
        np.clip(xyxy[:, [0, 2]], 0, self.img_size[1], out=xyxy[:, [0, 2]])
        np.clip(xyxy[:, [1, 3]], 0, self.img_size[0], out=xyxy[:, [1, 3]])
        return xyxy

    def sp_flatten(self, x):
        # (N, C, H, W) -> (N*H*W, C)
        return x.transpose(0, 2, 3, 1).reshape(-1, x.shape[1])

    def get_real_box(self, boxes):
        if len(boxes) == 0:
            return boxes
        boxes = boxes.copy()
        dw = self.letter_box_info.dw
        dh = self.letter_box_info.dh
        w_ratio = self.letter_box_info.w_ratio
        h_ratio = self.letter_box_info.h_ratio
        h, w = self.letter_box_info.org_shape[:2]

        boxes[:, 0] = (boxes[:, 0] - dw) / w_ratio
        boxes[:, 2] = (boxes[:, 2] - dw) / w_ratio
        boxes[:, 1] = (boxes[:, 1] - dh) / h_ratio
        boxes[:, 3] = (boxes[:, 3] - dh) / h_ratio

        np.clip(boxes[:, 0], 0, w - 1, out=boxes[:, 0])
        np.clip(boxes[:, 2], 0, w - 1, out=boxes[:, 2])
        np.clip(boxes[:, 1], 0, h - 1, out=boxes[:, 1])
        np.clip(boxes[:, 3], 0, h - 1, out=boxes[:, 3])

        return boxes

    def get_grid_and_stride(self, grid_h, grid_w):
        key = (grid_h, grid_w)
        if key not in self._grid_cache:
            col, row = np.meshgrid(
                np.arange(grid_w, dtype=np.float32),
                np.arange(grid_h, dtype=np.float32),
            )
            grid = np.stack((col, row), axis=0)[np.newaxis]  # (1,2,H,W)
            stride = np.array(
                [self.img_size[1] / grid_w, self.img_size[0] / grid_h],
                dtype=np.float32,
            ).reshape(1, 2, 1, 1)
            self._grid_cache[key] = (grid, stride)
        return self._grid_cache[key]

    def dfl(self, position):
        """Optimized Distribution Focal Loss decode."""
        n, c, h, w = position.shape
        mc = c // 4
        y = position.reshape(n, 4, mc, h, w)
        if y.dtype != np.float32:
            y = y.astype(np.float32)

        # In-place softmax along axis=2
        y -= y.max(axis=2, keepdims=True)
        np.exp(y, out=y)
        y /= y.sum(axis=2, keepdims=True)

        # Weighted sum via tensordot is faster than einsum for this shape
        proj = np.arange(mc, dtype=np.float32)
        return np.tensordot(y, proj, axes=([2], [0]))  # (n, 4, h, w)

    def limit_detections(self, boxes, classes, scores):
        if boxes is None or len(boxes) == 0:
            return self.empty_result()

        if self.max_det is None or self.max_det == 0 or len(scores) <= self.max_det:
            order = scores.argsort()[::-1]
            return boxes[order], classes[order], scores[order]

        order = scores.argsort()[::-1]
        boxes = boxes[order]
        classes = classes[order]
        scores = scores[order]

        if not self.keep_multi_class:
            return (
                boxes[:self.max_det],
                classes[:self.max_det],
                scores[:self.max_det],
            )

        unique_classes = np.unique(classes)

        if unique_classes.size <= 1:
            return (
                boxes[:self.max_det],
                classes[:self.max_det],
                scores[:self.max_det],
            )

        selected = []
        used = set()

        for cls in unique_classes:
            cls_inds = np.where(classes == cls)[0]
            if cls_inds.size > 0:
                selected.append(cls_inds[0])
                used.add(cls_inds[0])

                if len(selected) == self.max_det:
                    break

        if len(selected) > self.max_det:
            selected = sorted(
                selected,
                key=lambda i: scores[i],
                reverse=True
            )[:self.max_det]

        if len(selected) < self.max_det:
            for i in range(len(scores)):
                if i not in used:
                    selected.append(i)
                    used.add(i)

                    if len(selected) == self.max_det:
                        break

        selected = np.array(selected, dtype=np.int32)
        selected = selected[np.argsort(scores[selected])[::-1]]

        return boxes[selected], classes[selected], scores[selected]


    def post_process(self, input_data):
        pair_per_branch = len(input_data) // self.model_branch

        all_boxes = [None] * self.model_branch
        all_cls = [None] * self.model_branch

        for i in range(self.model_branch):
            idx = pair_per_branch * i
            b = self.box_process(input_data[idx])
            c = input_data[idx + 1]

            all_boxes[i] = self.sp_flatten(b)
            all_cls[i] = self.sp_flatten(c)

        boxes = np.concatenate(all_boxes, axis=0)
        cls_conf = np.concatenate(all_cls, axis=0)

        boxes, classes, scores = self.filter_boxes(boxes, cls_conf)

        if boxes.shape[0] == 0:
            return self.empty_result()

        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

        if not valid.all():
            boxes, classes, scores = boxes[valid], classes[valid], scores[valid]

            if boxes.shape[0] == 0:
                return self.empty_result()

        if self.pre_nms_topk is not None and boxes.shape[0] > self.pre_nms_topk:
            topk = np.argpartition(scores, -self.pre_nms_topk)[-self.pre_nms_topk:]
            boxes, classes, scores = boxes[topk], classes[topk], scores[topk]

        unique_classes = np.unique(classes)

        if unique_classes.size == 1:
            keep = self.nms_boxes(boxes, scores)

            if keep.size == 0:
                return self.empty_result()

            boxes = boxes[keep]
            classes = classes[keep]
            scores = scores[keep]

            order = scores.argsort()[::-1]
            boxes, classes, scores = boxes[order], classes[order], scores[order]

            return self.limit_detections(boxes, classes, scores)

        nboxes, nclasses, nscores = [], [], []

        for c in unique_classes:
            inds = np.where(classes == c)[0]
            keep = self.nms_boxes(boxes[inds], scores[inds])

            if keep.size > 0:
                nboxes.append(boxes[inds[keep]])
                nclasses.append(np.full(keep.size, c, dtype=classes.dtype))
                nscores.append(scores[inds[keep]])

        if not nboxes:
            return self.empty_result()

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        order = scores.argsort()[::-1]
        boxes, classes, scores = boxes[order], classes[order], scores[order]
        if self.max_det != 0:
            return self.limit_detections(boxes, classes, scores)
        return boxes, classes, scores


    def empty_result(self):
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )

    def end_to_end_inference(self, img: np.ndarray):
        """End-to-end inference pipeline."""

        padded = self.letter_box_preprc(img)

        model_out = self.run(padded)
        boxes, clss, conf = self.post_process(model_out)

        if boxes is not None and len(boxes) > 0:
            return self.get_real_box(boxes), clss, conf

        return self.empty_result()

    # def 


class InferenceClsRKNN:
    def __init__(
        self,
        model_path: str,
        img_size: tuple[int, int],
        cls_prob=0.3,
        npu_core=2,
    ):
        self.model_path = model_path
        self.img_size = img_size
        self.cls_prob = cls_prob
        self.npu_core = npu_core
        self.rknn_model = None
        self.setup_rknn_model()

    def setup_rknn_model(self):
        self.rknn_model = RKNNLite()
        ret = self.rknn_model.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")
        ret = self.rknn_model.init_runtime(core_mask=self.npu_core)
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime: {ret}")

    def preprocess(self, img, pad_color=(0,0,0)):
        h, w = img.shape[:2]
        scale = min(self.img_size[0] / h, self.img_size[1] / w)

        new_h = int(h * scale)
        new_w = int(w * scale)

        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = (self.img_size[0] - new_h) // 2
        pad_w = (self.img_size[1] - new_w) // 2

        letterboxed_img = cv2.copyMakeBorder(
            resized_img,
            pad_h, self.img_size[0] - new_h - pad_h,
            pad_w, self.img_size[1] - new_w - pad_w,
            cv2.BORDER_CONSTANT,
            value=pad_color,
        )
        return letterboxed_img[np.newaxis]
    
    def run(self, img: np.ndarray):
        clss = self.rknn_model.inference([img])
        if clss[0].max() > self.cls_prob:
            return  clss[0].argmax(), float(clss[0].max())