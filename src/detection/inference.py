

import cv2
import numpy as np
from rknnlite.api import RKNNLite




class InferenceRKNN:
    def __init__(self,
                 model_path: str = None,
                 img_size=(160, 160),
                 model_branch=3,
                 nms_thresh=0.6,
                 obj_thresh=0.5,
                 npu_core=0):
        self.model_path = model_path
        self.rknn_model = None
        self.mask_core_list = (RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2)
        self.npu_core = self.mask_core_list[npu_core]
        self.img_size = img_size

        self.nms_thresh = nms_thresh
        self.obj_thresh = obj_thresh
        self.model_branch = model_branch


    def letter_box_preprc(self,img: np.ndarray,
                   target_size: tuple = (160, 160),
                   pad_color: tuple=(0, 0, 0),
                   ) -> np.ndarray:
        """
        Resize and pad image while maintaining aspect ratio, preprocessing the input image.
        
        Args:
            img: Input image (numpy array)
            target_size: Target size (width, height) tuple
            pad_color: Padding color (BGR tuple)
        
        Returns:
            letterboxed_img: Padded and resized image
        """
        h, w = img.shape[:2]
        
        scale = min(target_size[0] / h, target_size[1] / w)
        
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        pad_h = (target_size[0] - new_h) // 2
        pad_w = (target_size[1] - new_w) // 2
        
        letterboxed_img = cv2.copyMakeBorder(
            resized_img, 
            pad_h, 
            target_size[0] - new_h - pad_h, 
            pad_w, 
            target_size[1] - new_w - pad_w, 
            cv2.BORDER_CONSTANT, 
            value=pad_color
        )
        
        return letterboxed_img


    def setup_rknn_model(self):
        """
        Setup the RKNN model.
        """
        self.rknn_model = RKNNLite()
        ret = self.rknn_model.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")
        ret = self.rknn_model.init_runtime(mask_core=self.npu_core)
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime: {ret}")
        

    def run(self, img: np.ndarray):
        """
        Run inference on the input image.
        
        Args:
            img: Input image (RGB format)
        
        Returns:
            detections: List of detected objects with bounding boxes and confidence scores
        """
        return self.model.inference(img)

    def release(self):
        self.model.release()
        self.model = None


    def filter_boxes(self, boxes: np.ndarray, box_conf: np.ndarray, box_cls_probs: np.ndarray):
        """Filter boxes with threshold - optimized."""
        box_conf = box_conf.ravel()
        cls_max_score = np.max(box_cls_probs, axis=-1)
        scores = cls_max_score * box_conf
        
        mask = scores >= self.obj_thresh
        
        return boxes[mask], np.argmax(box_cls_probs, axis=-1)[mask], scores[mask]

    def nms_boxes(self, boxes: np.ndarray, scores: np.ndarray):
        """Fast vectorized NMS."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Vectorized IoU computation
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[np.where(ovr <= self.nms_thresh)[0] + 1]
        
        return np.array(keep, dtype=np.int32)

    def box_process(self, position):
        """Optimized box processing."""
        grid_h, grid_w = position.shape[2:4]
        
        # Create grid once
        col = np.arange(0, grid_w).reshape(1, 1, 1, grid_w)
        row = np.arange(0, grid_h).reshape(1, 1, grid_h, 1)
        
        stride_h = self.img_size[1] // grid_h
        stride_w = self.img_size[0] // grid_w
        
        
        # Vectorized coordinate computation
        box_x1 = (col + 0.5 - position[:, 0:1, :, :]) * stride_w
        box_y1 = (row + 0.5 - position[:, 1:2, :, :]) * stride_h
        box_x2 = (col + 0.5 + position[:, 2:3, :, :]) * stride_w
        box_y2 = (row + 0.5 + position[:, 3:4, :, :]) * stride_h
        
        return np.concatenate((box_x1, box_y1, box_x2, box_y2), axis=1)

    def post_process(self, input_data):
        """Post-processing model output data."""
        pair_per_branch = len(input_data) // self.model_branch
        
        # Process all branches at once
        boxes_list = []
        clss_conf_list = []
        
        for i in range(self.model_branch):
            boxes_list.append(self.sp_flatten(self.box_process(input_data[pair_per_branch*i])))
            clss_conf_list.append(self.sp_flatten(input_data[pair_per_branch*i+1]))
        
        boxes = np.concatenate(boxes_list)
        clss_conf = np.concatenate(clss_conf_list)
        scores = np.ones((boxes.shape[0], 1), dtype=np.float32)
        
        # Filter boxes
        boxes, classes, scores = self.filter_boxes(boxes, scores, clss_conf)
        
        if len(boxes) == 0:
            return None, None, None
        
        # Per-class NMS with pre-allocated arrays
        unique_classes = np.unique(classes)
        keep_mask = np.zeros(len(boxes), dtype=bool)
        
        for c in unique_classes:
            mask = classes == c
            indices = np.where(mask)[0]
            keep_indices = self.nms_boxes(boxes[indices], scores[indices])
            keep_mask[indices[keep_indices]] = True
        
        return boxes[keep_mask], classes[keep_mask], scores[keep_mask]

    def end_to_end_inference(self, img: np.ndarray):
        """End-to-end inference pipeline."""
        preprocessed_img = self.letter_box_preprc(img, self.img_size)
        input_data = self.run(preprocessed_img)
        return self.post_process(input_data)
    