"""
doc
"""
import os
import time
from threading import Thread, Event, Lock
from datetime import datetime, timezone
from typing import Dict, Tuple, List
from copy import copy, deepcopy
from functools import wraps
import cv2

#import numpy as np
from detection.process import Detection
from cfg.config import CONFIG_DB_PATH, MAIN_DB_PATH#,IP_CAM_ADDRESS, PROCESS_FRAME_RATE
from cfg.getSDcardAdd import findSDcardAdd
from database.dbManager import DatabaseManager


class RunWorker():
    def __init__(self, frame_queue, processed_queue, killer, name = "RunWorker"):
        super().__init__(daemon=False, name=name)
        self.frame_queue = frame_queue
        self.processed_queue = processed_queue
        self.killer = killer
        self.region = ()
        self.detection = Detection()
        self.db_manager = DatabaseManager(db_name=MAIN_DB_PATH,
                                          confing_db=CONFIG_DB_PATH)
        self.lp_wanted_list = []
        self.det_dict = {}
        self.id_offset = 0
        self.sd_card_flag = False

    def run(self):
        i = 0
        update_interval = 15  # Seconds between wanted list updates
        last_update_time = time.time()
        # parent_save_path, det_img_path, undet_img_path = self._init_sd_card()
        parent_save_path = ""
        det_img_path = ""
        undet_img_path = ""
        self.db_manager.create_table()

        try:
            self.lp_wanted_list = self.db_manager.read_lp_alert()
        except Exception as e:
            print(f"Error fetching wanted list: {e}")
        self.id_offset = int(self.db_manager.get_last_id()) \
            if self.db_manager.get_last_id() else 0

        while not self.killer.is_stopped():





            # t1 = time.perf_counter()
            frame = self.frame_queue.get()
            if frame is object():
                break
            last_update_time = self._update_db_config(update_interval, last_update_time)
            annotated_frame = self._process_frame(frame, det_img_path, undet_img_path, "vid", i)     
            self.processed_queue.put(annotated_frame)
            # time.sleep(0.00001)
            i += 1
            # print(time.perf_counter() - t1)
        
        self.processed_queue.put(object())
        self._cleanup()
        print(f"[{self.name}] Stopped.")


    def _process_frame(self, frame, det_img_path: str, undet_img_path: str, cam_type: str, i):
        """
        doc
        """
        fans = False
        if self.region:
            h, w, _ = frame.shape
            points = (int(self.region[0]*w/100), int(self.region[1]*h/100),
                      int(self.region[2]*w/100), int(self.region[3]*h/100)) # x1y1x2y2 
            region_frame = frame[points[1]:points[3],points[0]:points[2]]
            fans = True

        else:
            region_frame = frame
            points = None
        # cv2.imwrite(f"nf/frame_{i}.jpg", region_frame)
        res = self.detection.process(region_frame, i, fans)
        # print(f'result of frame {i}: {res}')
        if res:
            self._update_det_dict(res, frame, det_img_path, undet_img_path, cam_type, self.region, points)
        else:
            self.det_dict.clear()

        annotated_frame = frame
        if points:
            for id_, (lp, _, box, _, state, _) in self.det_dict.items():
                if state != 1:
                    text = f"ID:{id_}-{lp}"
                    annotated_frame = self.annotate_boxes(annotated_frame,
                                                            (box[0][0]+points[0], box[0][1]+points[1]),
                                                            (box[1][0]+points[0], box[1][1]+points[1]), text=text)
            self.annotate_boxes(annotated_frame, (points[0], points[1]), (points[2],points[3]), color=True, thickness=8)
        else:
            for id_, (lp, _, box, _, state, _) in self.det_dict.items():
                if state != 1:
                    text = f"ID:{id_}-{lp}"
                    annotated_frame = self.annotate_boxes(frame = annotated_frame,
                                                        p1 = box[0], 
                                                        p2 = box[1], text=text)
        annotated_frame = cv2.resize(annotated_frame, (1600,1200))
        return annotated_frame

    def _update_db_config(self, update_interval: float, last_update_time: float) -> float:
        """Update the wanted list if the interval has elapsed."""
        current_time = time.time()
        if current_time - last_update_time >= update_interval:
            self.lp_wanted_list = self.db_manager.read_lp_alert()
            self.region = self.db_manager.read_region()
            return current_time
        return last_update_time

    def _update_det_dict(self,
        new_det_dict: Dict[int, Tuple[str, int, List[Tuple[int, int]], float, int, int]],
        frame, det_img_path: str, undet_img_path: str, cam_type: str, region: bool = False,
        points: List = None):
        """Update detection dictionary and database with new detections.
        Args:
            new_det_dict: Dictionary of detections with ID as key 
                and (license_plate, num_lp, box, prob, det_stat) as value.
            frame: Current video frame.
            det_img_path: Path for saving detected images.
            undet_img_path: Path for saving undetected images.
            cam_type: Source type ('vid' or 'cam').
        """
        tstamp = datetime.now(timezone.utc).isoformat()
        new_keys = set(new_det_dict.keys())
        current_keys = set(self.det_dict)
        lp_wanted_set = set(self.lp_wanted_list) if self.lp_wanted_list is not None else []
        global_ids = [id_ + self.id_offset for id_ in new_keys]

        db_exists = self.db_manager.batch_exists_id(global_ids)
        db_records = self.db_manager.batch_get_comp_record(global_ids)

        for id_ in new_keys:
            lp, num_lp, box, prob, det_stat, lp_width = new_det_dict[id_]
            # print(f"detection data {id_}:{new_det_dict[id_]}")
            global_id = id_ + self.id_offset
            in_db = db_exists.get(global_id, False)
            if len(lp_wanted_set):
                is_alert = lp in lp_wanted_set
            else:
                is_alert = False
            db_num_lp = db_records.get(global_id)[1] if in_db else 0
            db_lp = db_records.get(global_id)[0] if in_db else None
            db_lp_width = int(db_records.get(global_id)[2]) if in_db else 0
            db_lpchar_prob = float(db_records.get(global_id)[3]) if in_db else 0
            try:
                y1, y2 = box[0][1], box[1][1]
                x1, x2 = box[0][0], box[1][0]
                if not (0 <= y1 < y2 <= frame.shape[0] and 0 <= x1 < x2 <= frame.shape[1]):
                    print(f"Invalid ROI coordinates for ID {global_id}: {box}")
                    continue
                if region:
                    x1 += points[0]
                    y1 += points[1]
                    x2 += points[0]
                    y2 += points[1]
                roi = frame[y1:y2, x1:x2]
            except IndexError as e:
                print(f"ROI extraction failed for ID {global_id}: {e}")
                continue
            # print(f"db data:{id_, db_lp, db_num_lp, db_lp_width}")
            # save_path = None
            save_path_db = "/home/pi/car-detector/public/detected/"
            update_dict = True
            # remove_undet = False

            if not in_db and det_stat == 2:
                save_path_db = os.path.join(save_path_db, f"{global_id}.jpg")
                self.db_manager.insert(global_id, tstamp, tstamp, lp, str(num_lp), str(prob), 
                                    None, None, None, None, is_alert, cam_type, False, 0)

            elif not in_db and det_stat == 3:
                save_path_db = os.path.join(save_path_db, f"{global_id}.jpg")
                self.db_manager.insert(global_id, tstamp, tstamp, lp, str(num_lp), str(prob), 
                                    None, None, None, None, is_alert, cam_type, False, str(lp_width))

            elif in_db and (int(db_num_lp)<num_lp<8 or
                            (det_stat==3 and 
                            ((lp_width>db_lp_width) or (int(db_num_lp)<num_lp) or (prob>db_lpchar_prob))
                            )
                            ):
                save_path_db = os.path.join(save_path_db, f"{global_id}.jpg")
                self.db_manager.modify(global_id, tstamp, lp, num_lp, str(prob), is_alert, str(lp_width))
            else:
                update_dict = False

            if save_path_db and update_dict:
                cv2.imwrite(save_path_db, roi)

            if update_dict:
                self.det_dict[id_] = [lp, num_lp, box, prob, det_stat, lp_width]
            else:
                self.det_dict[id_] = [db_lp, db_num_lp, box, db_lpchar_prob, det_stat, db_lp_width]
        for id_ in current_keys - new_keys:
            del self.det_dict[id_]

    def _cleanup(self):
        """Clean up resources."""
        del self.detection

    @staticmethod
    def annotate_boxes(frame, p1, p2, text=None, color:bool=None, thickness=2):
        """
        annotation of output images: draw car boxes with license plate text
        """
        if color:
            cv2.rectangle(frame, p1, p2, (0, 255, 0), thickness)
            return frame
        cv2.rectangle(frame, p1, p2, (255, 255, 255), thickness)
        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 1.5, thickness)[0]
            text_x, text_y = p1[0], p1[1]
        
            cv2.rectangle(frame, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), (0, 0, 255), -1)

            cv2.putText(frame, text, (text_x, text_y), font, 1.5, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
        return frame

    def _init_sd_card(self) -> Tuple[str, str, str]:
        sd_card_path = findSDcardAdd().findMount()
        if not sd_card_path:
            raise IOError("SD card slot not detected. Please check the SD card slot.")

        self.sd_card_flag = True
        tstamp = datetime.now().strftime("/%Y-%m-%d_%H-%M-%S/")
        parent_path = sd_card_path + tstamp
        save_img_path = parent_path + "img/"
        undet_img_path = save_img_path + "undetected/"
        det_img_path = save_img_path + "detected/"

        for path in (parent_path, save_img_path, undet_img_path, det_img_path):
            os.makedirs(path, exist_ok=True)

        return parent_path, det_img_path, undet_img_path

    @staticmethod
    def timing(fn):
        """
        doc
        """
        @wraps(fn)
        def wrapper(*args, **kwargs):
            st = time.time()
            res = fn(*args, **kwargs)
            et = time.time()
            print(f"{fn.__name__} time {et - st:.2f}")
            return res
        return wrapper

class GracefulKiller:
    def __init__(self):
        self._kill = Event()

    def stop(self):
        self._kill.set()
    
    def is_stopped(self):
        return self._kill.is_set()