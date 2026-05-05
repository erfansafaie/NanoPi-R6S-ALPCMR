"""
doc
"""
import time
from typing import Dict, Tuple, List
from collections import OrderedDict
from cfg.config import MODEL_PATH_LIST
from .model_infer import ModelInference
import cv2

from src.detection.inference import InferenceRKNN
from src.detection.tracker.bot_sort import BoTSORT

class Detection:
    """
    doc
    """
    # Class-level constants for labels and sub-labels
    _LABELS = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5',
        6: '6', 7: '7', 8: '8', 9: '9', 10: 'alef',
        11: 'b', 12: 'je', 13: 'lam', 14: 'mim', 15: 'noon',
        16: 'qaf', 17: 'vav', 18: 'h', 19: 'ye', 20: 'dal',
        21: 'sin', 22: 'sad', 23: 'malol', 24: 'te',
        25: 'ta', 26: 'ein', 27: 'diplomat', 28: 'siyasi',
        29: 'p', 31: 'the', 32: 'ze', 33: 'shin', 34: 'fe',
        35: 'kaf', 36: 'gaf'
    }

    _SUB_CHAR_LABEL = {17, 11, 26, 34, 24, 21, 28, 32, 16,
                       22, 15, 23, 35, 18, 10, 20, 33, 25,
                       27, 29, 19, 13, 14, 31, 12, 36}

    _SUB_CHAR_LABEL_NUM = set(range(10))

    def __init__(self, v_model_path: str = MODEL_PATH_LIST[0],
                 lpd_model_path: str = MODEL_PATH_LIST[1],
                 lpr_model_path: str = MODEL_PATH_LIST[2]
        ):
        self._model_inference = ModelInference(v_model_path, lpd_model_path, lpr_model_path)
        self.car_det_model = InferenceRKNN(model_path=v_model_path,
                                           img_size=(192, 320),
                                           model_branch=3,
                                           nms_thresh=0.6,
                                           obj_thresh=0.5,
                                           npu_core=0)
        self.lpd_model = InferenceRKNN(model_path=lpd_model_path,
                                      img_size=(160, 160),
                                      model_branch=2,
                                      nms_thresh=0.7,
                                      obj_thresh=0.5,
                                      npu_core=1)
        self.lpr_model = InferenceRKNN(model_path=lpr_model_path,
                                      img_size=(160, 160),
                                      model_branch=2,
                                      nms_thresh=0.7,
                                      obj_thresh=0.5,
                                      npu_core=1)
        self.car_color_model = None
        self.car_type_model = None

        self.car_tracker = BoTSORT()


    def _extract_frame(self, frame, box: List[float]) -> Tuple:
        x1, y1, x2, y2 = map(int, box)
        
        return frame[y1:y2, x1:x2], ((x1, y1), (x2, y2))

    def _format_license_plate(self, chars: List[str]) -> str:
        return f"{chars[0]}{chars[1]}_{chars[2]}_{chars[3]}{chars[4]}{chars[5]}_{chars[6]}{chars[7]}"

    def process(self, frame, i, fans) -> Dict[int, List]:
        """
        doc
        """

        detection_dict: Dict[int, Tuple] = OrderedDict()
        min_v_size = 120 if fans else 160
        v_boxex, v_clss, v_scores = self.car_det_model.end_to_end_inference(frame) # box shape x1 y1 x2 y2
        v_ids = self.car_tracker.update(v_boxex, v_clss, v_scores)
        if len(v_ids) == 0:
            return detection_dict

        for car_id, box, v_cls in zip(v_ids, v_boxex, v_clss):

            vframe, box_cord = self._extract_frame(frame, box)

            det_stat = 1
            if v_cls == 3:
                det_stat = 2
                detection_dict[int(car_id)] = [None, 0, box_cord, 0.0, det_stat, 0]
                # lp_boxes, _ = self._model_inference.infer_lpd_model(vframe)
                # if lp_boxes.size>0:
                #     det_stat = 2
                #     detection_dict[int(vp_id)] = [None, 0, box_cord, 0.0, det_stat, int(lp_boxes[0][2] - lp_boxes[0][0])]
                # else:
                #     det_stat = 1
                #     detection_dict[int(vp_id)] = [None, 0, box_cord, 0.0, det_stat, 0]
                #     continue
 
            if vframe.shape[0] <= min_v_size or vframe.shape[1] <= min_v_size:
                continue

            lp_boxes = self.lpd_model.end_to_end_inference(vframe)

            if len(lp_boxes) == 0:
                continue

            det_stat = 2
            lp_frame, _ = self._extract_frame(vframe, lp_boxes[0])
            if lp_frame.shape[0] <= 30 or lp_frame.shape[1] <= 75:
                detection_dict[int(car_id)] = [None, 0, box_cord, 0.0, det_stat, lp_frame.shape[1]]
                continue

            lpchar_data, lp_char_num = self.lpr_model.end_to_end_inference(lp_frame)
            # print(f"\n***lp rec time: {time.time()-t1:.3f}s")
            if lpchar_data.size==0:
                continue
            if lp_char_num==0:
                continue
            # print(f"frame {i} numlp: {num_lp}")

            if (lp_char_num == 8 and
                all(lpchar_data[j][5] in self._SUB_CHAR_LABEL_NUM \
                    for j in (0, 1, 3, 4, 5, 6, 7)) and
                lpchar_data[2][5] in self._SUB_CHAR_LABEL):
                det_stat = 3
                prob = round(sum(lpchar_data[:, 4])/8, 4)
                license_plate = [self._LABELS[int(char[5])] for char in lpchar_data]
                # print(license_plate)
                formatted_lp = self._format_license_plate(license_plate)
                detection_dict[int(vp_id)] = [formatted_lp, lp_char_num, box_cord, prob, det_stat, lp_frame.shape[1]]
            else:
                # license_plate = [self._LABELS[int(char[5])] for char in lp_box_char]
                # print(license_plate)
                detection_dict[int(vp_id)] = [None, lp_char_num, box_cord, 0.0, det_stat, lp_frame.shape[1]]

        return detection_dict

    def license_plate_prc(self, v_img):
        pass

    def color_type_prc(self, v_img):

