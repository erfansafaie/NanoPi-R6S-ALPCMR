# import os
# import sys
# import time
# import threading
# from types import SimpleNamespace

# import cv2
# import numpy as np

# sys.path.insert(0, str(os.getcwd()))

# from src.detection.inference import InferenceDetRKNN
# from src.detection.tracker.bot_sort import BOTSORT


# args = SimpleNamespace(
#     track_high_thresh=0.5,
#     track_low_thresh=0.3,
#     new_track_thresh=0.6,
#     track_buffer=30,
#     match_thresh=0.8,
#     fuse_score=True,
#     class_aware = True,
#     gate_center_dist = 400.0,   # tune per resolution
#     gate_size_ratio = 4.0
# )


# class Results:
#     def __init__(self, tlbr, conf, cls):
#         self.tlbr = np.asarray(tlbr, dtype=np.float32)
#         self.conf = np.asarray(conf, dtype=np.float32)
#         self.cls = np.asarray(cls, dtype=np.int32)


# # Shared state
# latest_frame = None           # newest raw frame from capture
# latest_display_frame = None   # newest annotated frame for display

# frame_lock = threading.Lock()
# display_lock = threading.Lock()
# stop_event = threading.Event()


# def create_vid_gst_pipeline(vid_src):
#     return (
#         f"filesrc location={vid_src} ! "
#         f"qtdemux ! h265parse ! mppvideodec ! rgaconvert ! "
#         f"video/x-raw, format=BGR, width=1920, height=1080 ! "
#         f"appsink max-buffers=20 sync=true"
#     )


# def get_color(track_id):
#     # Stable color per ID
#     np.random.seed(track_id % 2**16)
#     color = tuple(int(x) for x in np.random.randint(50, 255, size=3))
#     return color


# def draw_track(frame, track):
#     """
#     Expected track format:
#     [x1, y1, x2, y2, track_id, score, cls_id]
#     """
#     if track is None or len(track) < 7:
#         return

#     x1, y1, x2, y2, track_id, score, cls_id = track[:7]
#     x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#     track_id = int(track_id)
#     cls_id = int(cls_id)
#     score = float(score)

#     color = get_color(track_id)

#     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#     label = f"ID:{track_id} CLS:{cls_id} CONF:{score:.2f}"
#     (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)

#     y_text = max(th + 4, y1 - 6)
#     cv2.rectangle(frame, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 2), color, -1)
#     cv2.putText(
#         frame,
#         label,
#         (x1 + 3, y_text - 3),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.55,
#         (0, 0, 0),
#         2,
#         cv2.LINE_AA,
#     )


# def draw_info(frame, fps, num_tracks):
#     cv2.putText(
#         frame,
#         f"FPS: {fps:.2f}",
#         (20, 30),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.8,
#         (0, 255, 255),
#         2,
#         cv2.LINE_AA,
#     )
#     cv2.putText(
#         frame,
#         f"Tracks: {num_tracks}",
#         (20, 65),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.8,
#         (0, 255, 255),
#         2,
#         cv2.LINE_AA,
#     )


# def capture_worker(cap):
#     global latest_frame

#     while not stop_event.is_set():
#         ret, frame = cap.read()
#         if not ret:
#             print("VideoCapture cannot read frame")
#             stop_event.set()
#             break

#         with frame_lock:
#             latest_frame = frame


# def inference_worker(vehicle_model, tracker):
#     global latest_frame, latest_display_frame

#     prev_time = time.perf_counter()

#     while not stop_event.is_set():
#         frame = None

#         with frame_lock:
#             if latest_frame is not None:
#                 frame = latest_frame.copy()
#                 latest_frame = None  # consume latest frame

#         if frame is None:
#             time.sleep(0.005)
#             continue

#         t1 = time.perf_counter()

#         try:
#             boxes, clss, conf = vehicle_model.end_to_end_inference(frame)
#             res = Results(boxes, conf, clss)
#             tracks = tracker.update(res, frame)
#         except Exception as e:
#             print(f"Inference/tracking error: {e}")
#             continue

#         annotated = frame.copy()

#         num_tracks = 0
#         if tracks is not None:
#             for trk in tracks:
#                 draw_track(annotated, trk)
#             num_tracks = len(tracks)

#         now = time.perf_counter()
#         fps = 1.0 / max(now - prev_time, 1e-6)
#         prev_time = now

#         infer_time = time.perf_counter() - t1
#         draw_info(annotated, fps, num_tracks)

#         cv2.putText(
#             annotated,
#             f"Infer: {infer_time*1000:.1f} ms",
#             (20, 100),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (0, 255, 255),
#             2,
#             cv2.LINE_AA,
#         )

#         with display_lock:
#             latest_display_frame = annotated

#         # print(f"Detection + Tracking time: {infer_time:.4f}s | tracks={num_tracks}")


# def display_worker(window_name="Tracking"):
#     global latest_display_frame

#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

#     while not stop_event.is_set():
#         frame = None

#         with display_lock:
#             if latest_display_frame is not None:
#                 frame = latest_display_frame.copy()

#         if frame is not None:
#             cv2.imshow(window_name, frame)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             stop_event.set()
#             break

#         time.sleep(0.001)

#     cv2.destroyAllWindows()


# def test_vehicle_detector():
#     model_path = "/home/pi/NanoPi-R6S-ALPCMR/src/models/vd_640_v11.rknn"

#     vehicle_model = InferenceDetRKNN(
#         model_path=model_path,
#         img_size=(192, 320),
#         use_dfl=True,
#         npu_core=1,
#         max_det=8,
#     )

#     tracker = BOTSORT(args, frame_rate=30)

#     cap = cv2.VideoCapture(create_vid_gst_pipeline("3.mp4"), cv2.CAP_GSTREAMER)
#     # Alternative if needed:
#     # cap = cv2.VideoCapture("3.mp4")

#     if not cap.isOpened():
#         print("Cannot open video")
#         return

#     t_cap = threading.Thread(target=capture_worker, args=(cap,), daemon=True)
#     t_inf = threading.Thread(target=inference_worker, args=(vehicle_model, tracker), daemon=True)
#     t_disp = threading.Thread(target=display_worker, daemon=True)

#     t_cap.start()
#     t_inf.start()
#     t_disp.start()

#     try:
#         while not stop_event.is_set():
#             time.sleep(0.1)
#     except KeyboardInterrupt:
#         stop_event.set()

#     t_cap.join(timeout=1.0)
#     t_inf.join(timeout=1.0)
#     t_disp.join(timeout=1.0)

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     test_vehicle_detector()


import os
import sys
import time
import threading
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(os.getcwd()))

from src.detection.inference import InferenceDetRKNN
from src.detection.tracker.bot_sort import BOTSORT


args = SimpleNamespace(
    track_high_thresh=0.6,
    track_low_thresh=0.3,
    new_track_thresh=0.4,
    track_buffer=10,
    match_thresh=0.7,
    fuse_score=True,
)

class Results:
    def __init__(self, tlbr, conf, cls):
        self.tlbr = np.asarray(tlbr, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.int32)


latest_frame = None
frame_lock = threading.Lock()
stop_event = threading.Event()


WIDTH = 1920
HEIGHT = 1080
FPS = 30


def create_vid_gst_pipeline(vid_src):
    return (
        f"filesrc location={vid_src} ! "
        f"qtdemux ! h265parse ! mppvideodec ! rgaconvert ! "
        f"video/x-raw, format=BGR, width={WIDTH}, height={HEIGHT} ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


def create_writer_gst_pipeline(output_path, fps=30):
    return (
        f"appsrc ! "
        f"video/x-raw,format=BGR,width={WIDTH},height={HEIGHT},framerate={fps}/1 ! "
        f"queue ! "
        f"rgaconvert ! "
        f"video/x-raw,format=NV12,width={WIDTH},height={HEIGHT},framerate={fps}/1 ! "
        f"mpph264enc ! "
        f"h264parse ! "
        f"mp4mux ! "
        f"filesink location={output_path} sync=false"
    )


def get_color(track_id):
    np.random.seed(int(track_id) % 65535)
    return tuple(int(x) for x in np.random.randint(50, 255, size=3))


def draw_track(frame, track):
    """
    Expected tracker output:
    [x1, y1, x2, y2, track_id, score, cls_id]
    """

    if track is None or len(track) < 7:
        return

    x1, y1, x2, y2, track_id, score, cls_id = track[:7]

    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    track_id = int(track_id)
    cls_id = int(cls_id)
    score = float(score)

    color = get_color(track_id)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"ID:{track_id} CLS:{cls_id} CONF:{score:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_w, text_h = text_size

    label_y = max(y1, text_h + 10)

    cv2.rectangle(
        frame,
        (x1, label_y - text_h - 8),
        (x1 + text_w + 8, label_y + baseline),
        color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (x1 + 4, label_y - 4),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_info(frame, fps, infer_ms, num_tracks):
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Infer+Track: {infer_ms:.1f} ms",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Tracks: {num_tracks}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def capture_worker(cap):
    global latest_frame

    while not stop_event.is_set():
        ret, frame = cap.read()

        if not ret:
            print("VideoCapture finished or cannot read frame")
            stop_event.set()
            break

        with frame_lock:
            latest_frame = frame


def inference_worker(vehicle_model, tracker, writer):
    global latest_frame

    prev_time = time.perf_counter()
    frame_count = 0

    while not stop_event.is_set():
        frame = None

        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
                latest_frame = None

        if frame is None:
            time.sleep(0.001)
            continue

        t1 = time.perf_counter()

        boxes, clss, conf = vehicle_model.end_to_end_inference(frame)
        now = time.perf_counter()
        res = Results(boxes, conf, clss)
        tracks = tracker.update(boxes, conf, clss)
        fps = 1.0 / max(now - t1, 1e-6)
        infer_ms = (now - t1) * 1000.0
        annotated = frame.copy()

        num_tracks = 0

        if tracks is not None:
            for trk in tracks:
                
                draw_track(annotated, trk)

            num_tracks = len(tracks)



        

        draw_info(
            frame=annotated,
            fps=fps,
            infer_ms=infer_ms,
            num_tracks=num_tracks,
        )

        writer.write(annotated)

        # frame_count += 1

        # print(
        #     f"Frame: {frame_count} | "
        #     f"Infer+Track: {infer_ms:.2f} ms | "
        #     f"FPS: {fps:.2f} | "
        #     f"Tracks: {num_tracks}"
        # )


def test_vehicle_detector():
    model_path = "/home/pi/NanoPi-R6S-ALPCMR/src/models/vd_640_v11.rknn"
    input_video = "3.mp4"
    output_video = "annotated_output.mp4"

    vehicle_model = InferenceDetRKNN(
        model_path=model_path,
        img_size=(192, 320),
        use_dfl=True,
        npu_core=1,
        max_det=4,
    )

    tracker = BOTSORT(args, frame_rate=FPS)

    cap_pipeline = create_vid_gst_pipeline(input_video)
    cap = cv2.VideoCapture(cap_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Cannot open input video")
        return

    writer_pipeline = create_writer_gst_pipeline(output_video, FPS)

    writer = cv2.VideoWriter(
        writer_pipeline,
        cv2.CAP_GSTREAMER,
        0,
        FPS,
        (WIDTH, HEIGHT),
        True,
    )

    if not writer.isOpened():
        print("Cannot open GStreamer VideoWriter")
        cap.release()
        return

    print(f"Recording annotated video to: {output_video}")

    t_cap = threading.Thread(
        target=capture_worker,
        args=(cap,),
        daemon=True,
    )

    t_inf = threading.Thread(
        target=inference_worker,
        args=(vehicle_model, tracker, writer),
        daemon=True,
    )

    t_cap.start()
    t_inf.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping...")
        stop_event.set()

    t_cap.join(timeout=2.0)
    t_inf.join(timeout=2.0)

    cap.release()
    writer.release()

    print(f"Saved annotated output video: {output_video}")


if __name__ == "__main__":
    test_vehicle_detector()
