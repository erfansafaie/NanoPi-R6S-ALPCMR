"""
doc
"""
import os
import time
from queue import Queue, Empty
from threading import Thread, Event, Lock
from datetime import datetime, timezone
from typing import Dict, Tuple, List
from copy import copy, deepcopy
from functools import wraps

from PIL import Image

#import numpy as np
from detection.process import Detection
from cfg.config import CONFIG_DB_PATH, MAIN_DB_PATH#,IP_CAM_ADDRESS, PROCESS_FRAME_RATE
from cfg.getSDcardAdd import findSDcardAdd
from src.database.db_manager import DatabaseManager


class RunWorker():
    def __init__(
            self,
            front_frame_queue,
            rear_frame_queue,
            result_queue,
            killer,
        ):
        self.front_frame_queue = front_frame_queue
        self.rear_frame_queue = rear_frame_queue
        self.result_queue = result_queue

        self.killer = killer
        self.region = ()

        self.detection = Detection(self.killer)
        self.db_data_handler = DatabaseDataHandler(self.killer, self.result_queue)

        self.det_dict = {}
        self.id_offset = 0
        self.sd_card_flag = False


        self.front_cam_thread = Thread(target=self.detection.process_front_cam, args=(self.front_frame_queue, 120), daemon=True)
        self.rear_cam_thread = Thread(target=self.detection.process_rear_cam, args=(self.rear_frame_queue, 120), daemon=True)
        self.lp_thread = Thread(target=self.detection.process_lp_queue, daemon=True)
        self.attr_thread = Thread(target=self.detection.process_color_model, daemon=True)
        self.aggregate_thread = Thread(target=self.detection.aggregation_data, args=(self.result_queue,), daemon=True)
        self.database_thread = Thread(target=self.db_data_handler.parse_check_res, daemon=True)

    def run(self):
        self.front_cam_thread.start()
        time.sleep(0.1)
        self.rear_cam_thread.start()
        time.sleep(0.1)
        self.lp_thread.start()
        time.sleep(0.1)
        self.attr_thread.start()
        time.sleep(0.1)
        self.aggregate_thread.start()
        time.sleep(0.1)
        self.database_thread.start()
    
    def stop(self):
        self.front_cam_thread.join()
        self.rear_cam_thread.join()
        self.lp_thread.join()
        self.attr_thread.join()
        self.aggregate_thread.join()
        self.database_thread.join()


class DatabaseDataHandler:
    def __init__(self, killer, data_queue):
        self.data_queue = data_queue
        self.killer = killer
        self.save_image_pathm = "/home/pi/car-detector/public/detected"
        self.lp_alert_set = {}

        self.update_alert_time = 30
        self.last_update_alert_time = 0
    

    def parse_check_res(self):
        self.db_manager = DatabaseManager(db_name=MAIN_DB_PATH, confing_db=CONFIG_DB_PATH)
        self.lp_alert_set = self.db_manager.read_lp_alert()
        while not self.killer.is_stopped():
            try:
                data = self.data_queue.get(timeout=0.1)
                if not data:
                    time.sleep(0.08)

                self.unpack_process_data(data)
                t = time.time()
                if (t - self.last_update_alert_time) > self.update_alert_time:
                    self.lp_alert_set = self.db_manager.read_lp_alert()
                    self.last_update_alert_time = t
            except Empty:
                time.sleep(0.8)
        self.db_manager.close_conn()

    def convert_eptime2datetime(self, eptime):
        return datetime.fromtimestamp(eptime, timezone.utc).isoformat()
    
    def unpack_process_data(self, data):
        parsed_data_indices = {}
        for index, det in enumerate(data):
            uid = det[4]
            parsed_data_indices[uid] = index 

        uid_list = list(parsed_data_indices.keys())
        existing_db_dict = self.db_manager.get_batch_record(uid_list)

        inserts = []
        for uid, index in parsed_data_indices.items():
            current_data = data[index]
            current_tuple = (
                uid, self.convert_eptime2datetime(current_data[2]),  
                current_data[7], current_data[8], current_data[10], current_data[11],
                current_data[12], current_data[13], current_data[14], current_data[15],
                current_data[1], False
            )
            
            if uid not in existing_db_dict:
                if current_tuple[5] == 8:
                    inserts.append(current_tuple)
                    self.save_image(current_data[5], uid)
            else:
                update, lp_mod, is_better_data = self.is_data_better(current_tuple, existing_db_dict[uid])
                if update:
                    self.db_manager.modify(is_better_data)
                    if lp_mod:
                        self.save_image(current_data[5], uid)

        self.db_insert(inserts)


    def is_data_better(self, new_data, db_data):
        is_alert = False

        if (new_data[3] > float(db_data[2])) or (new_data[4] > float(db_data[3])):
            tstamp = new_data[1]
            lp_char = new_data[2]
            prob = new_data[3]
            lp_width = new_data[4]
            lp_num = new_data[5]
            lp_mod = True
            if self.lp_alert_set:
                if lp_char in self.lp_alert_set:
                    is_alert = True
                else:
                    is_alert = False
            else:
                is_alert = False
        else:
            tstamp = db_data[0]
            lp_char = db_data[1]
            prob = db_data[2]
            lp_width = db_data[3]
            lp_num = db_data[4]
            lp_mod = False

        if (new_data[7] > float(db_data[6])):
            color_mod = True
            color = new_data[6]
            color_prob = new_data[7]
        else:
            color_mod = False
            color = db_data[5]
            color_prob = db_data[6]

        if (new_data[9] > float(db_data[8])):
            model_mod = True
            model = new_data[8]
            model_prob = new_data[9]
        else:
            model_mod = False
            model = db_data[7]
            model_prob = db_data[8]

        if not (lp_mod or color_mod or model_mod):
            return False, False, None
        uid = new_data[0]
        return True, lp_mod, (
            tstamp, 
            lp_char, 
            prob, 
            lp_width,
            lp_num,
            color, 
            color_prob, 
            model, 
            model_prob, 
            is_alert, 
            uid
        )

    def save_image(self, vframe_data, img_fname):
        img = Image.fromarray(vframe_data)
        img.save(os.path.join(self.save_image_pathm, img_fname + '.jpg'))
    
    def db_insert(self, records):
        is_alert = False
        for r in records:
            if self.lp_alert_set:
                if r[2] in self.lp_alert_set:
                    r[-1] = True
            self.db_manager.insert(r)

        # frame_id_
        # update_interval = 15  # Seconds between wanted list updates
        # last_update_time = time.time()
        # # parent_save_path, det_img_path, undet_img_path = self._init_sd_card()
        # parent_save_path = ""
        # det_img_path = ""
        # undet_img_path = ""
        # self.db_manager.create_table()

        # try:
        #     self.lp_wanted_list = self.db_manager.read_lp_alert()
        # except Exception as e:
        #     print(f"Error fetching wanted list: {e}")
        # self.id_offset = int(self.db_manager.get_last_id()) \
        #     if self.db_manager.get_last_id() else 0

        # while not self.killer.is_stopped():





        #     # t1 = time.perf_counter()
        #     frame = self.frame_queue.get()
        #     if frame is object():
        #         break
        #     last_update_time = self._update_db_config(update_interval, last_update_time)
        #     annotated_frame = self._process_frame(frame, det_img_path, undet_img_path, "vid", i)     
        #     # time.sleep(0.00001)
        #     i += 1
        #     # print(time.perf_counter() - t1)
        
        # self._cleanup()
        # print(f"[{self.name}] Stopped.")


    # def _process_frame(self, frame, det_img_path: str, undet_img_path: str, cam_type: str, i):
    #     """
    #     doc
    #     """
    #     fans = False
    #     if self.region:
    #         h, w, _ = frame.shape
    #         points = (int(self.region[0]*w/100), int(self.region[1]*h/100),
    #                   int(self.region[2]*w/100), int(self.region[3]*h/100)) # x1y1x2y2 
    #         region_frame = frame[points[1]:points[3],points[0]:points[2]]
    #         fans = True

    #     else:
    #         region_frame = frame
    #         points = None
    #     # cv2.imwrite(f"nf/frame_{i}.jpg", region_frame)
    #     res = self.detection.process(region_frame, i, fans)
    #     # print(f'result of frame {i}: {res}')
    #     if res:
    #         self._update_det_dict(res, frame, det_img_path, undet_img_path, cam_type, self.region, points)
    #     else:
    #         self.det_dict.clear()

    #     annotated_frame = frame
    #     if points:
    #         for id_, (lp, _, box, _, state, _) in self.det_dict.items():
    #             if state != 1:
    #                 text = f"ID:{id_}-{lp}"
    #                 annotated_frame = self.annotate_boxes(annotated_frame,
    #                                                         (box[0][0]+points[0], box[0][1]+points[1]),
    #                                                         (box[1][0]+points[0], box[1][1]+points[1]), text=text)
    #         self.annotate_boxes(annotated_frame, (points[0], points[1]), (points[2],points[3]), color=True, thickness=8)
    #     else:
    #         for id_, (lp, _, box, _, state, _) in self.det_dict.items():
    #             if state != 1:
    #                 text = f"ID:{id_}-{lp}"
    #                 annotated_frame = self.annotate_boxes(frame = annotated_frame,
    #                                                     p1 = box[0], 
    #                                                     p2 = box[1], text=text)
    #     annotated_frame = cv2.resize(annotated_frame, (1600,1200))
    #     return annotated_frame

    # def _update_db_config(self, update_interval: float, last_update_time: float) -> float:
    #     """Update the wanted list if the interval has elapsed."""
    #     current_time = time.time()
    #     if current_time - last_update_time >= update_interval:
    #         self.lp_wanted_list = self.db_manager.read_lp_alert()
    #         self.region = self.db_manager.read_region()
    #         return current_time
    #     return last_update_time

    # def _update_det_dict(self,
    #     new_det_dict: Dict[int, Tuple[str, int, List[Tuple[int, int]], float, int, int]],
    #     frame, det_img_path: str, undet_img_path: str, cam_type: str, region: bool = False,
    #     points: List = None):
    #     """Update detection dictionary and database with new detections.
    #     Args:
    #         new_det_dict: Dictionary of detections with ID as key 
    #             and (license_plate, num_lp, box, prob, det_stat) as value.
    #         frame: Current video frame.
    #         det_img_path: Path for saving detected images.
    #         undet_img_path: Path for saving undetected images.
    #         cam_type: Source type ('vid' or 'cam').
    #     """
    #     tstamp = datetime.now(timezone.utc).isoformat()
    #     new_keys = set(new_det_dict.keys())
    #     current_keys = set(self.det_dict)
    #     lp_wanted_set = set(self.lp_wanted_list) if self.lp_wanted_list is not None else []
    #     global_ids = [id_ + self.id_offset for id_ in new_keys]

    #     db_exists = self.db_manager.batch_exists_id(global_ids)
    #     db_records = self.db_manager.batch_get_comp_record(global_ids)

    #     for id_ in new_keys:
    #         lp, num_lp, box, prob, det_stat, lp_width = new_det_dict[id_]
    #         # print(f"detection data {id_}:{new_det_dict[id_]}")
    #         global_id = id_ + self.id_offset
    #         in_db = db_exists.get(global_id, False)
    #         if len(lp_wanted_set):
    #             is_alert = lp in lp_wanted_set
    #         else:
    #             is_alert = False
    #         db_num_lp = db_records.get(global_id)[1] if in_db else 0
    #         db_lp = db_records.get(global_id)[0] if in_db else None
    #         db_lp_width = int(db_records.get(global_id)[2]) if in_db else 0
    #         db_lpchar_prob = float(db_records.get(global_id)[3]) if in_db else 0
    #         try:
    #             y1, y2 = box[0][1], box[1][1]
    #             x1, x2 = box[0][0], box[1][0]
    #             if not (0 <= y1 < y2 <= frame.shape[0] and 0 <= x1 < x2 <= frame.shape[1]):
    #                 print(f"Invalid ROI coordinates for ID {global_id}: {box}")
    #                 continue
    #             if region:
    #                 x1 += points[0]
    #                 y1 += points[1]
    #                 x2 += points[0]
    #                 y2 += points[1]
    #             roi = frame[y1:y2, x1:x2]
    #         except IndexError as e:
    #             print(f"ROI extraction failed for ID {global_id}: {e}")
    #             continue
    #         # print(f"db data:{id_, db_lp, db_num_lp, db_lp_width}")
    #         # save_path = None
    #         save_path_db = "/home/pi/car-detector/public/detected/"
    #         update_dict = True
    #         # remove_undet = False

    #         if not in_db and det_stat == 2:
    #             save_path_db = os.path.join(save_path_db, f"{global_id}.jpg")
    #             self.db_manager.insert(global_id, tstamp, tstamp, lp, str(num_lp), str(prob), 
    #                                 None, None, None, None, is_alert, cam_type, False, 0)

    #         elif not in_db and det_stat == 3:
    #             save_path_db = os.path.join(save_path_db, f"{global_id}.jpg")
    #             self.db_manager.insert(global_id, tstamp, tstamp, lp, str(num_lp), str(prob), 
    #                                 None, None, None, None, is_alert, cam_type, False, str(lp_width))

    #         elif in_db and (int(db_num_lp)<num_lp<8 or
    #                         (det_stat==3 and 
    #                         ((lp_width>db_lp_width) or (int(db_num_lp)<num_lp) or (prob>db_lpchar_prob))
    #                         )
    #                         ):
    #             save_path_db = os.path.join(save_path_db, f"{global_id}.jpg")
    #             self.db_manager.modify(global_id, tstamp, lp, num_lp, str(prob), is_alert, str(lp_width))
    #         else:
    #             update_dict = False

    #         if save_path_db and update_dict:
    #             cv2.imwrite(save_path_db, roi)

    #         if update_dict:
    #             self.det_dict[id_] = [lp, num_lp, box, prob, det_stat, lp_width]
    #         else:
    #             self.det_dict[id_] = [db_lp, db_num_lp, box, db_lpchar_prob, det_stat, db_lp_width]
    #     for id_ in current_keys - new_keys:
    #         del self.det_dict[id_]

    # def _cleanup(self):
    #     """Clean up resources."""
    #     del self.detection

    # @staticmethod
    # def annotate_boxes(frame, p1, p2, text=None, color:bool=None, thickness=2):
    #     """
    #     annotation of output images: draw car boxes with license plate text
    #     """
    #     if color:
    #         cv2.rectangle(frame, p1, p2, (0, 255, 0), thickness)
    #         return frame
    #     cv2.rectangle(frame, p1, p2, (255, 255, 255), thickness)
    #     if text:
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         text_size = cv2.getTextSize(text, font, 1.5, thickness)[0]
    #         text_x, text_y = p1[0], p1[1]
        
    #         cv2.rectangle(frame, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), (0, 0, 255), -1)

    #         cv2.putText(frame, text, (text_x, text_y), font, 1.5, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    #     return frame

    # def _init_sd_card(self) -> Tuple[str, str, str]:
    #     sd_card_path = findSDcardAdd().findMount()
    #     if not sd_card_path:
    #         raise IOError("SD card slot not detected. Please check the SD card slot.")

    #     self.sd_card_flag = True
    #     tstamp = datetime.now().strftime("/%Y-%m-%d_%H-%M-%S/")
    #     parent_path = sd_card_path + tstamp
    #     save_img_path = parent_path + "img/"
    #     undet_img_path = save_img_path + "undetected/"
    #     det_img_path = save_img_path + "detected/"

    #     for path in (parent_path, save_img_path, undet_img_path, det_img_path):
    #         os.makedirs(path, exist_ok=True)

    #     return parent_path, det_img_path, undet_img_path

    # @staticmethod
    # def timing(fn):
    #     """
    #     doc
    #     """
    #     @wraps(fn)
    #     def wrapper(*args, **kwargs):
    #         st = time.time()
    #         res = fn(*args, **kwargs)
    #         et = time.time()
    #         print(f"{fn.__name__} time {et - st:.2f}")
    #         return res
    #     return wrapper

class GracefulKiller:
    def __init__(self):
        self._kill = Event()

    def stop(self):
        self._kill.set()
    
    def is_stopped(self):
        return self._kill.is_set()