"""
doc
"""
import time
from typing import Dict, Tuple, List
from collections import OrderedDict
from queue import Queue

from src.cfg.config import MODEL_PATH_LIST
from src.detection.model_infer import ModelInference
from src.detection.inference import InferenceDetRKNN
from src.detection.tracker.bot_sort import BOTSORT

class Detection:
    """
    doc
    """
    # Class-level constants for labels and sub-labels


    def __init__(
            self,
            v_model_path: str = MODEL_PATH_LIST[0],
            lpd_model_path: str = MODEL_PATH_LIST[1],
            lpr_model_path: str = MODEL_PATH_LIST[2]
        ):
        self._model_inference = ModelInference(v_model_path, lpd_model_path, lpr_model_path)

        self.car_color_model = None
        self.car_type_model = None

        self.running = False
        self.lp_queue = Queue(maxsize=16)
        self.attr_queue = Queue(maxsize=16)
        self.results = Queue(maxsize=32)

    def _extract_frame(self, frame, box: List[float]) -> Tuple:
        x1, y1, x2, y2 = map(int, box)
        
        return frame[y1:y2, x1:x2], ((x1, y1), (x2, y2))

    def _format_license_plate(self, chars: List[str]) -> str:
        return f"{chars[0]}{chars[1]}_{chars[2]}_{chars[3]}{chars[4]}{chars[5]}_{chars[6]}{chars[7]}"

    def process_camera_frame(self, front_cam_frame, rear_cam_frame, fans):
        """
        doc
        """
        detection_dict: Dict[int, Tuple] = OrderedDict()
        min_v_size = 120 if fans else 160
        front_tracks = self._model_inference.track_front_cam(front_cam_frame) # box shape x1 y1 x2 y2
        rear_tracks = self._model_inference.track_rear_cam(rear_cam_frame)
        for track_data in front_tracks:
            if track_data[4] != 3:
                v_frame = self._extract_frame(track_data[:4])
                if v_frame.shape[0] <= min_v_size or v_frame.shape[1] <= min_v_size:
                    continue
                self.lp_queue.put((track_data[-1], "F", v_frame))
                self.attr_queue.put((track_data[-1], "F", v_frame))
        for track_data in rear_tracks:
            if track_data[4] != 3:
                v_frame = self._extract_frame(track_data[:4])
                if v_frame.shape[0] <= min_v_size or v_frame.shape[1] <= min_v_size:
                    continue
                self.lp_queue.put((track_data[-1], "R", v_frame))
                self.attr_queue.put((track_data[-1], "R", v_frame))

    def process_lp_queue(self):

        while self.running:
            if self.lp_queue.not_empty():
                v_frame, cam, v_id = self.lp_queue.get()
            else:
                continue
            lp_box, _, lp_conf = self._model_inference.infer_lpd_model(v_frame)
            lp_frame = self._extract_frame(v_frame, lp_box)
            if lp_frame.shape[0] <= 30 or lp_frame.shape[1] <= 75:
                                                                    #TODO update the result queue
                continue
            lp_chars, prob = self._model_inference.process_lpr(lp_frame)
            if not lp_chars:
                continue                                                
                                                                    #TODO update the result queue


    def process_color_type(self):

        while self.running:
            if self.lp_queue.not_empty():
                v_frame, cam, v_id = self.attr_queue.get()
            else:
                continue
            (v_color, prob_v_color), (v_model, prob_v_model) = self._model_inference.color_type_prc()
                                                                    #TODO update the result queue

    def aggregation_data(self):
        pass