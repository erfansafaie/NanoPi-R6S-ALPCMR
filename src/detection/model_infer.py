from dataclasses import dataclass
import time

import numpy as np

from src.detection.inference import InferenceDetRKNN, InferenceClsRKNN
from src.detection.tracker.bot_sort import BOTSORT


@dataclass
class TrackerSet:
    track_high_thresh: float = 0.4,
    track_low_thresh: float = 0.2,
    new_track_thresh: float = 0.4,
    track_buffer: int = 15,
    match_thresh: float = 0.8,
    fuse_score: bool = True,


class ModelInference:
    """
    class of models inference
    """
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

    _COLOR_LABELS = ('Black', 'Blue', 'Brown', 'Dark Green', 'Dark Red', 'Gold',
                'Gray', 'Green', 'Orange', 'Red', 'Silver', 'White', 'Yellow')
    _MODEL_LABELS = ('Arisun', 'Atlas', 'Bahman Fidelity', 'Baic Sabrina',
                    'Chery Tiggo 5', 'Dena', 'Fownix FX', 'H30Cross', 'Haima S7', 'Jac J4',
                    'Jac J5', 'Jac S3', 'Jac S5', 'KMC T8', 'KaraMazdaPickup', 'Kia Cerato',
                    'Kia Mohave', 'MVM315H', 'MVMX22', 'NeissanVanet', 'Peugeot_206',
                    'Peugeot_206_SD', 'Peugeot_207', 'Peugeot_405', 'Peugeot_Pars',
                    'PeykanSavari', 'PeykanVanet', 'Pride_Nasim', 'Pride111', 'Pride131',
                    'Pride132', 'Pride141', 'Pride151', 'Quik', 'Renault_L90', 'Renault_Sandro',
                    'RenaultPK', 'RioSD', 'Runna', 'Saina', 'Samand', 'SamandSoren', 'Shahin',
                    'Tara', 'Tiba', 'Tiba2', 'Xantia')

    def __init__(
            self,
            v_model_path,
            lpd_model_path,
            lpr_model_path
            ):
        """
        """
        self._tracker_args = TrackerSet()
        self._front_cam_tracker = BOTSORT(self.tracker_args)
        self._rear_cam_tracker = BOTSORT(self.tracker_args)

        self._vd_model = InferenceDetRKNN(
            model_path=v_model_path,
            img_size=(192,320),
            model_branch=3,
            obj_thresh=0.4,
            nms_thresh=0.6,
            pre_nms_topk=100,
            max_det=6,
            keep_multi_class=False,
            reg_max=16,
            npu_core=0,
            use_dfl=True
        )
        self._lpd_model = InferenceDetRKNN(
            model_path=lpd_model_path,
            img_size=(160,160),
            model_branch=2,
            obj_thresh=0.4,
            nms_thresh=0.6,
            pre_nms_topk=16,
            max_det=1,
            keep_multi_class=False,
            reg_max=1,
            npu_core=1,
            use_dfl=False
        )
        self._lpr_model = InferenceDetRKNN(
            model_path=lpr_model_path,
            img_size=(160,160),
            model_branch=2,
            obj_thresh=0.4,
            nms_thresh=0.6,
            pre_nms_topk=64,
            max_det=8,
            keep_multi_class=False,
            reg_max=1,
            npu_core=1,
            use_dfl=False
        )

        self._color_model = InferenceClsRKNN(
            model_path="",
            img_size=(128,128),
            cls_prob=0.8,
            npu_core=2
        )
        self._type_model = InferenceClsRKNN(
            model_path="",
            img_size=(128,128),
            cls_prob=0.8,
            npu_core=2
        )

    def track_front_cam(self, image):
        boxes, clss, scores  = self._vd_model.end_to_end_inference(image)
        return self._front_cam_tracker(boxes, clss, scores)

    def track_rear_cam(self, image):
        boxes, clss, scores  = self._vd_model.end_to_end_inference(image)
        return self._rear_cam_tracker(boxes, clss, scores)      

    def infer_lpd_model(self, image):
        return self._lpd_model.end_to_end_inference(image)

    def infer_lpr_model(self, image):
        boxes, clss, scores = self._lpd_model.end_to_end_inference(image)
        if boxes.size>0:
            order = boxes[:,0].argsort()
            return boxes[order], clss[order], scores[order]
        return boxes, clss, scores

    def remove_dup(
            self, boxes: np.ndarray,
            clss: np.ndarray,
            scores: np.ndarray,
            x_pixel_thresh: float = 3.0,
            expected_chars: int = 8
        ):
        """
        Docstring for remove_dup
        
        :param det: Description
        :param x_pixel: Description
        :param expect_char: Description
        """
        n = len(boxes)
        i = 0
        keep_indc = []
        
        while i < n:
            j = i + 1
            while j<n and (boxes[j,0] - boxes[i, 0] <= x_pixel_thresh):
                j += 1
            if j - i > 1:
                best_local_idx = np.argmax(scores[i:j])
                keep_indc.append(i + best_local_idx)
            else:
                keep_indc.append(i)
            i = j
        filtered_boxes = boxes[keep_indc]
        filtered_scores = scores[keep_indc]
        filtered_classes = clss[keep_indc]
        if len(filtered_boxes) > expected_chars:
            top_n_indc = np.argpartition(filtered_scores, -expected_chars)[-expected_chars:]
            top_n_indc = top_n_indc[np.argsort(filtered_boxes[top_n_indc, 0])]

            filtered_boxes = filtered_boxes[top_n_indc]
            filtered_scores = filtered_scores[top_n_indc]
            filtered_classes = filtered_classes[top_n_indc]

        return filtered_boxes, filtered_classes, filtered_scores, len(filtered_boxes)

    def format_license_plate(self, chars) -> str:
        return f"{chars[0]}{chars[1]}_{chars[2]}_{chars[3]}{chars[4]}{chars[5]}_{chars[6]}{chars[7]}"

    def process_lpr(self,image):
        lpr_res_raw = self.infer_lpr_model(image)
        lpr_res = self.remove_dup(lpr_res_raw)
        if (                    
            lpr_res[3] == 8 and 
            all(lpr_res[1][j] in self._SUB_CHAR_LABEL_NUM for j in (0, 1, 3, 4, 5, 6, 7)) and
            lpr_res[1][2] in self._SUB_CHAR_LABEL
        ):
            prob = round(sum(lpr_res[2])/8, 4)
            lp_text = [self._LABELS[int(lpr_res[1][i])] for i in range(8)]
            return self.format_license_plate(lp_text), prob
        else:
            return None, 0.0
    
    def process_color_type(self, img: np.ndarray):
        img = self._color_model.preprocess(img)
        v_color = self._color_model.run(img)
        v_model = self._type_model.run(img)
        return self._COLOR_LABELS[v_color[0]], v_color[1], self._MODEL_LABELS[v_model[0]], v_model[1]
