"""
doc
"""
import time
import json
from typing import Dict, Any, Tuple, List, Optional
from collections import deque
from queue import Queue, Empty, Full
from dataclasses import dataclass

from src.cfg.config import MODEL_PATH_LIST
from src.detection.model_infer import ModelInference

BBox = Tuple[int, int, int, int]
CamID = str
VehicleID = str
FrameID = str
MAX_FRAMES_PER_VEHICLE = 5


class Detection:
    """
    doc
    """
    (IDX_FRAME, IDX_CAM, IDX_TSTAMP, IDX_VID, IDX_UID, IDX_VFRAME, IDX_BBOX, 
     IDX_LP_CHARS, IDX_LP_PROB, IDX_LP_BOX, IDX_LP_WIDTH, IDX_LP_NUM,
     IDX_V_COLOR, IDX_COLOR_PROB, IDX_V_MODEL, IDX_MODEL_PROB) = range(16)

    def __init__(
            self,
            killer,
            v_model_path: str = MODEL_PATH_LIST[0],
            lpd_model_path: str = MODEL_PATH_LIST[1],
            lpr_model_path: str = MODEL_PATH_LIST[2],
            color_model_path: str = MODEL_PATH_LIST[3],
            model_model_path: str = MODEL_PATH_LIST[4]
        ):
        self._model_inference = ModelInference(
            v_model_path,
            lpd_model_path,
            lpr_model_path,
            color_model_path,
            model_model_path)

        self.car_color_model = None
        self.car_type_model = None

        self.killer = killer
        self.running = False
        self.lp_queue = Queue(maxsize=16)
        self.attr_queue = Queue(maxsize=16)
        self.aggregate_queue = Queue(maxsize=32)

        # self.vehicle_history_manager = VehicleHistoryManager()

        self.pending_merges = {}
        self.history = {}
        self.start_frame_id = None

    def __del__(self):
        pass

    def _extract_frame(self, frame, box: Tuple[float]) -> Tuple:
        x1, y1, x2, y2 = map(int, box)
        
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def _format_license_plate(self, chars: List[str]) -> str:
        return f"{chars[0]}{chars[1]}_{chars[2]}_{chars[3]}{chars[4]}{chars[5]}_{chars[6]}{chars[7]}"

    def process_front_cam(self, front_queue: Queue, min_size):
        while not self.killer.is_stopped():
            try:
                frame, cam_id, frame_id, tstamp = front_queue.get(timeout=0.05)
            except Empty:
                continue
            if frame is None:
                continue
            # t1 = time.perf_counter()
            front_tracks = self._model_inference.track_front_cam(frame)
            # print("front----tracks:", front_tracks)
            # print('f*'*10, time.perf_counter() - t1)
            if not len(front_tracks)>0:
                continue
            for track_data in front_tracks:
                v_frame, bbox = self._extract_frame(frame, track_data[:4])
                if v_frame.shape[0] <= min_size or v_frame.shape[1] <= min_size:
                    continue
                item = (
                    frame_id,
                    cam_id,
                    tstamp,
                    int(track_data[-2]),
                    track_data[-1],
                    v_frame, bbox
                )
                self.put_drop_oldest(self.lp_queue, item)
                self.put_drop_oldest(self.attr_queue, item)

    def process_rear_cam(self, rear_queue: Queue, min_size):
        """
        doc
        """
        while not self.killer.is_stopped():
            try:
                frame, cam_id, frame_id, tstamp = rear_queue.get(timeout=0.05)
            except Empty:
                continue
            if frame is None:
                continue
            # t1 = time.perf_counter()
            rear_tracks = self._model_inference.track_rear_cam(frame)
            # print("rear----tracks:", rear_tracks)

            # print('r*'*10, time.perf_counter() - t1)

            if not len(rear_tracks)>0:
                continue
            for track_data in rear_tracks:
                if track_data[4] != 3:
                    v_frame, bbox = self._extract_frame(frame, track_data[:4])
                    if v_frame.shape[0] <= min_size or v_frame.shape[1] <= min_size:
                        continue
                    item = (
                        frame_id,
                        cam_id,
                        tstamp,
                        int(track_data[-2]),
                        track_data[-1],
                        v_frame, bbox
                    )
                    self.put_drop_oldest(self.lp_queue, item)
                    self.put_drop_oldest(self.attr_queue, item)

    def process_lp_queue(self):
        """
        
        """
        while not self.killer.is_stopped():
            try:
                frame_id, cam_id, tstamp, v_id, unique_id, v_frame, bbox = self.lp_queue.get(timeout=0.05)
            # t1 = time.perf_counter()
                lp_box, clss, lp_conf = self._model_inference.infer_lpd_model(v_frame)
            # print('*'*30, time.perf_counter() - t1)
            
                if not lp_box.size>0:
                    res = ("lp", (frame_id, cam_id, tstamp, v_id, unique_id, v_frame, bbox, None, 0.0, None, 0, 0))
                    self.put_drop_oldest(self.aggregate_queue, res)
                    continue

                lp_frame, lp_box = self._extract_frame(v_frame, lp_box[0])

                if lp_frame.shape[0] <= 29 or lp_frame.shape[1] <= 74:
                    res = ("lp", (frame_id, cam_id, tstamp, v_id, unique_id, v_frame, bbox, None, 0.0, lp_box, lp_box[2]-lp_box[0], 0))
                    self.put_drop_oldest(self.aggregate_queue, res)
                    continue

                # t1 = time.perf_counter()
                lp_chars, lp_chars_prob, lp_num = self._model_inference.process_lpr(lp_frame)
                # print('*'*40, time.perf_counter() - t1)

                if not lp_chars:
                    res = ("lp", (frame_id, cam_id, tstamp, v_id, unique_id, v_frame, bbox, None, 0.0, lp_box, lp_box[2]-lp_box[0], lp_num))
                    self.put_drop_oldest(self.aggregate_queue, res)
                    continue

                res = ("lp", (frame_id, cam_id, tstamp, v_id, unique_id, v_frame, bbox, lp_chars, lp_chars_prob, lp_box, lp_box[2]-lp_box[0], lp_num))
                
                self.put_drop_oldest(self.aggregate_queue, res)

            except Empty:
                time.sleep(0.02)

    def process_color_model(self):
        """
        frame_id, track_id, color, color_prob, model, model_prob
        """
        while not self.killer.is_stopped():
            try:
                frame_id, cam_id, tstamp, v_id, unique_id, v_frame, bbox = self.attr_queue.get(timeout=0.05)

                v_color, prob_v_color, v_model, prob_v_model = self._model_inference.process_color_model(v_frame)
                res = ("attr", (frame_id, cam_id, tstamp, v_id, unique_id, v_frame, bbox, v_color, prob_v_color, v_model, prob_v_model))
                self.put_drop_oldest(self.aggregate_queue, res)
            except Empty:
                time.sleep(0.02)

    def aggregation_data(self, result_queue: Queue):
        i = 0
        # with open("json_data.jsonl", "a", encoding="utf-8") as f:
        while not self.killer.is_stopped():
            try:
                res_data = self.aggregate_queue.get(timeout=0.05)
                ready, best_data = self._process_aggregate_result(res_data)
                if ready:
                    self.put_drop_oldest(result_queue, best_data)
            except Empty:
                time.sleep(0.02)

        remaining_data = self.flush_remaining()
        if remaining_data:
            self.put_drop_oldest(result_queue, remaining_data)


    def put_drop_oldest(self, q: Queue, item: tuple):
        try:
            q.put_nowait(item)
        except Full:
            try:
                q.get_nowait()
            except Empty:
                pass
            try:
                q.put_nowait(item)
            except Full:
                pass



            # if i == 0:
            #     fc_last = res_data[1][2]
            #     i +=1
            # fc_next = res_data[1][2]
            # if fc_last != fc_next:
            # # print(res_data[0], res_data[1][:4], res_data[1][5:])
            #     res_updt = self.vehicle_history_manager.get_all_histories()
            #     self.write_json(res_updt, f)
            #     fc_last = fc_next
            # self.vehicle_history_manager.update_history_from_res(res_data)

    def _process_aggregate_result(self, res_data):
        data_type, payload = res_data
        
        frame_id, cam_id, tstamp, v_id = payload[0], payload[1], payload[2], payload[3]
        
        merge_key = (frame_id, v_id, cam_id)
        
        if self.start_frame_id is None:
            self.start_frame_id = frame_id

        pair = self.pending_merges.setdefault(merge_key, [None, None])
        
        if data_type == "lp":
            pair[0] = payload
        else:
            pair[1] = payload

        if pair[0] is not None and pair[1] is not None:
            lp_p, attr_p = pair
            
            complete_data = [
                lp_p[0], lp_p[1], lp_p[2], lp_p[3], lp_p[4], lp_p[5], lp_p[6],  # Common (0-6)
                lp_p[7], lp_p[8], lp_p[9], lp_p[10], lp_p[11],                  # LP (7-11)
                attr_p[7], attr_p[8], attr_p[9], attr_p[10]                     # Attr (11-14)
            ]
            
            del self.pending_merges[merge_key]

            history_key = (v_id, cam_id)
            if history_key not in self.history:
                self.history[history_key] = complete_data
            else:
                self._update_best_data(self.history[history_key], complete_data)
        if frame_id - self.start_frame_id >= 10:
            best_data_to_flush = [tuple(v) for v in self.history.values()]
            
            self.history.clear()
            self._cleanup_old_pending_merges(frame_id)
            self.start_frame_id = frame_id
            return True, best_data_to_flush

        return False, None

    def _update_best_data(self, current_best, new_data):
        """
        In-place modification of current_best list using the index constants.
        """
        if (new_data[self.IDX_LP_PROB] > current_best[self.IDX_LP_PROB]): 
            
           # or (new_data[self.IDX_LP_WIDTH] > current_best[self.IDX_LP_WIDTH])): 
           # -------------------------------------------------------------------------------------
           # TODO: test lp width and aspect ratio to improve lp accuracy 
           # -------------------------------------------------------------------------------------

            current_best[self.IDX_VFRAME] = new_data[self.IDX_VFRAME]
            current_best[self.IDX_BBOX] = new_data[self.IDX_BBOX]
            current_best[self.IDX_LP_CHARS] = new_data[self.IDX_LP_CHARS]
            current_best[self.IDX_LP_PROB] = new_data[self.IDX_LP_PROB]
            current_best[self.IDX_LP_BOX] = new_data[self.IDX_LP_BOX]
            current_best[self.IDX_LP_WIDTH] = new_data[self.IDX_LP_WIDTH]
            current_best[self.IDX_LP_NUM] = new_data[self.IDX_LP_NUM]

        if new_data[self.IDX_COLOR_PROB] > current_best[self.IDX_COLOR_PROB]:
            current_best[self.IDX_V_COLOR] = new_data[self.IDX_V_COLOR]
            current_best[self.IDX_COLOR_PROB] = new_data[self.IDX_COLOR_PROB]

        if new_data[self.IDX_MODEL_PROB] > current_best[self.IDX_MODEL_PROB]:
            current_best[self.IDX_V_MODEL] = new_data[self.IDX_V_MODEL]
            current_best[self.IDX_MODEL_PROB] = new_data[self.IDX_MODEL_PROB]


    def _cleanup_old_pending_merges(self, current_frame_id, threshold=10):
        # List comprehension to find stale keys avoids RuntimeError during dict iteration
        stale_keys = [k for k in self.pending_merges if current_frame_id - k[0] > threshold]
        for k in stale_keys:
            del self.pending_merges[k]


    def flush_remaining(self):
        """
        Forces a flush of any complete data left in history 
        when closing the pipeline, ignoring the 5-frame threshold.
        """
        if not self.history:
            return None
            
        best_data_to_flush = [tuple(v) for v in self.history.values()]
        
        # Clean up
        self.history.clear()
        self.pending_merges.clear()
        self.start_frame_id = None
        
        return best_data_to_flush
    

'''

    def write_json(self, data, file_obj):
        json_ready_data = {}

        for key, records in data.items():
            # Convert tuple key to string
            new_key = f"{key[0]}_{key[1]}"

            json_ready_records = []
            for record in records:
                new_record = {}

                for k, v in record.items():
                    # Convert tuples to lists for JSON
                    if isinstance(v, tuple):
                        new_record[k] = list(v)
                    else:
                        new_record[k] = v

                json_ready_records.append(new_record)

            json_ready_data[new_key] = json_ready_records


        json.dump(json_ready_data, file_obj, ensure_ascii=False, indent=4)
        file_obj.write("\n\n")
        file_obj.flush()



class VehicleDataUpdate:
    def __init__(self):
        pass

    def _get_vehicle



class VehicleHistoryManager:
    def __init__(self, max_history=5, inactive_window=5, cleanup_interval=5):
        self.max_history = max_history
        self.inactive_window = inactive_window
        self.cleanup_interval = cleanup_interval
        self.global_result_counter = 0
        self.history_map = {}

    def _get_vehicle(self, cam_id, v_id):
        key = (cam_id, v_id)
        vehicle = self.history_map.get(key)
        if vehicle is None:
            vehicle = {
                "last_seen": self.global_result_counter,
                "records": deque(maxlen=self.max_history),
                "index": {}
            }
            self.history_map[key] = vehicle
        return vehicle

    def _new_record(self, frame_id, v_frame, bbox):
        return {
            "frame_id": frame_id,
            # "v_frame": v_frame,
            "bbox": bbox,
            "lp_chars": None,
            "lp_chars_prob": None,
            "lp_box": None,
            "v_color": None,
            "prob_v_color": None,
            "v_model": None,
            "prob_v_model": None,
        }

    def _get_or_create_record(self, vehicle, frame_id, v_frame, bbox):
        index = vehicle["index"]
        rec = index.get(frame_id)
        if rec is not None:
            return rec

        records = vehicle["records"]
        if len(records) == self.max_history:
            old = records[0]
            index.pop(old["frame_id"], None)

        rec = self._new_record(frame_id, v_frame, bbox)
        records.append(rec)
        index[frame_id] = rec
        return rec

    def _cleanup_inactive(self):
        now = self.global_result_counter
        dead_keys = [
            key for key, vehicle in self.history_map.items()
            if now - vehicle["last_seen"] >= self.inactive_window
        ]
        for key in dead_keys:
            del self.history_map[key]

    def update_history_from_res(self, res_data):
        self.global_result_counter += 1
        kind, data = res_data

        if kind == "lp":
            frame_id, cam_id, _, v_id, v_frame, bbox, lp_chars, lp_chars_prob, lp_box = data
            vehicle = self._get_vehicle(cam_id, v_id)
            rec = self._get_or_create_record(vehicle, frame_id, v_frame, bbox)

            # rec["v_frame"] = v_frame
            rec["bbox"] = bbox
            rec["lp_chars"] = lp_chars
            rec["lp_chars_prob"] = lp_chars_prob
            rec["lp_box"] = lp_box
            vehicle["last_seen"] = self.global_result_counter

        elif kind == "attr":
            frame_id, cam_id, _, v_id, v_frame, bbox, v_color, prob_v_color, v_model, prob_v_model = data
            vehicle = self._get_vehicle(cam_id, v_id)
            rec = self._get_or_create_record(vehicle, frame_id, v_frame, bbox)

            # rec["v_frame"] = v_frame
            rec["bbox"] = bbox
            rec["v_color"] = v_color
            rec["prob_v_color"] = prob_v_color
            rec["v_model"] = v_model
            rec["prob_v_model"] = prob_v_model
            vehicle["last_seen"] = self.global_result_counter

        else:
            return

        if self.global_result_counter % self.cleanup_interval == 0:
            self._cleanup_inactive()

    def get_vehicle_history(self, cam_id, v_id):
        vehicle = self.history_map.get((cam_id, v_id))
        return [] if vehicle is None else list(vehicle["records"])

    def get_all_histories(self):
        return {
            key: list(vehicle["records"])
            for key, vehicle in self.history_map.items()
        }

'''