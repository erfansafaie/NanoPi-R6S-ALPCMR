

import cv2
import numpy as np
from rknnlite.api import RKNNLite




class InferenceRKNN:
    def __init__(self, img_size=(160, 160), model_branch=3, nms_thresh=0.6, obj_thresh=0.5):
        self.rknn_model = None
        self.maskPcore = (RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2)
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


    def setup_rknn_model(self, model_path: str):
        """
        Setup the RKNN model.
        """
        self.rknn_model = RKNNLite()
        ret = self.rknn_model.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")
        ret = self.model.init_runtime(mask_core=RKNNLite.NPU_CORE_AUTO)
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
        """
        Filter boxes with threshold.
        """
        box_conf = box_conf.reshape(-1)
        # candidate, class_num = box_cls_probs.shape

        cls_max_score = np.max(box_cls_probs, axis=-1)
        classes = np.argmax(box_cls_probs, axis=-1)

        _class_pos = np.where(cls_max_score* box_conf >= self.obj_thresh)
        scores = (cls_max_score* box_conf)[_class_pos]

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

        return boxes, classes, scores

    def nms_boxes(self, boxes: np.ndarray, scores: np.ndarray):
        """
        Suppress non-maximal boxes.
        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:

            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        keep = np.array(keep)
        return keep

    def box_process(self, position):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([self.img_size[1]//grid_h, self.img_size[0]//grid_w]).reshape(1,2,1,1)

        position = dfl(position)
        box_xy  = grid +0.5 -position[:,0:2,:,:]
        box_xy2 = grid +0.5 +position[:,2:4,:,:]
        xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

        return xyxy

    def post_process(self, input_data):
        boxes, scores, clss_conf = [], [], []
        pair_per_branch = len(input_data)//self.model_branch
        for i in range(self.model_branch):
            boxes.append(self.box_process(input_data[pair_per_branch*i]))
            clss_conf.append(input_data[pair_per_branch*i+1])
            scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))


        boxes = [self.sp_flatten(_v) for _v in boxes]
        clss_conf = [self.sp_flatten(_v) for _v in clss_conf]
        scores = [self.sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        clss_conf = np.concatenate(clss_conf)
        scores = np.concatenate(scores)

        boxes, classes, scores = self.filter_boxes(boxes, scores, clss_conf)

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = self.nms_boxes(b, s)

            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c[keep])
                nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores