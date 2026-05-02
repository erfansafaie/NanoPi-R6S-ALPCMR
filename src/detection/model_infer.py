
from ultralytics import YOLO
import numpy as np
from .ocsort import OCSort
import time


class ModelInference:
    """
    class of 
    """

    def __init__(self, v_model_path, lpd_model_path, lpr_model_path):
        """
        """
        self.vd_model = YOLO(v_model_path, task="detect")
        self.lpd_model = YOLO(lpd_model_path, task="detect")
        self.lpr_model = YOLO(lpr_model_path, task="detect")
        # self.oc_tracker = OCSort(det_thresh=0.33, max_age=5, min_hits=3, iou_threshold=0.7, delta_t=3, asso_func="giou", inertia=0.2, use_byte=True)

    
    def infer_vd_model(self, image):

        # t1 = time.perf_counter()
        vp_res = self.vd_model.track(image, conf=0.38, iou=0.6,
                    half=True,max_det=6, imgsz=(256,320),#(192,320), 
                    stream=False, verbose=False, persist=True,
                    tracker="/home/pi/lp/lpMain/detection/tracker.yaml")#, classes=[2,3,5,7])

        boxes = vp_res[0].boxes.xyxy.cpu().numpy()
        clss = vp_res[0].boxes.cls.cpu().numpy()
        if vp_res[0].boxes.id is not None:
            v_id = vp_res[0].boxes.id.cpu().numpy()
        else:
            v_id = np.array([])
        # print(time.perf_counter()-t1)

        # t1 = time.time()
        # id = np.array([])
        # vp_res = self.vd_model.predict(image, conf=0.3, iou=0.7,
        #             half=True,max_det=8, imgsz=(256,320),#(192,320), 
        #             stream=False, verbose=False)
        # boxes = vp_res[0].boxes.xyxy.cpu().numpy()
        # confs = vp_res[0].boxes.conf.cpu().numpy()[:, np.newaxis]
        # clss = vp_res[0].boxes.cls.cpu().numpy()
        # if len(boxes)>0:
        #     detections = np.hstack((boxes, confs))  # [N, 5]: x1,y1,x2,y2,conf
        # else:
        #     detections = np.empty((0, 5))

        # tracks = self.oc_tracker.update(detections, image.shape, image.shape)
        # if tracks.shape[0] != 0:
        #     id = [s[-1] for s in tracks]
        #     boxes = [s[:-1] for s in tracks]
        # print(time.time()-t1)

        del vp_res
        return v_id, boxes, clss
         
    def infer_lpd_model(self,image):
        lpd_res = self.lpd_model.predict(image, conf=0.6,
                    half=True, imgsz=160, verbose=False)
        boxes = lpd_res[0].boxes.xyxy.cpu().numpy()
        del lpd_res
        return boxes

    def infer_lpr_model(self, image):
        lpr_res = self.lpr_model.predict(image, conf=0.51, max_det=8,
                    half=True, imgsz=160, verbose=False, iou=0.6)
        # _, indc = lpr_res[0].boxes.data[:,0].sort()
        # boxes = lpr_res[0].boxes.data[indc].cpu().numpy()
        # del lpr_res
        return lpr_res[0].boxes.data[lpr_res[0].boxes.data[:,0].sort()[1]].cpu().numpy()

    def remove_dup(self, det_chars, x_pixel_thresh=3, expected_chars=8):
        """
        Docstring for remove_dup
        
        :param det: Description
        :param x_pixel: Description
        :param expect_char: Description
        """
        n = len(det_chars)
        # if len(det_chars) <= 8:
        #     return det_chars
        i = 0
        keep_indc = []
        while i < n:
            j = i + 1
            while j<n and (det_chars[j,0] - det_chars[i, 0] <= x_pixel_thresh):
                j += 1
            if j - i > 1:
                best_local_idx = np.argmax(det_chars[i:j, 4])
                keep_indc.append(i + best_local_idx)
            else:
                keep_indc.append(i)
            i = j
        filtered_det = det_chars[keep_indc]
        if len(filtered_det) > expected_chars:
            top_n_indc = np.argpartition(filtered_det[:, 4], -expected_chars)[-expected_chars:]
            top_n_indc = top_n_indc[np.argsort(filtered_det[top_n_indc, 0])]
            filtered_det = filtered_det[top_n_indc]

        return filtered_det, len(filtered_det)

    def process_lpr(self,image):
        lpr_res = self.infer_lpr_model(image)
        return self.remove_dup(lpr_res)