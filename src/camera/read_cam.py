# #
# import subprocess
# import queue
# import time
# import base64
# from threading import Thread
# import cv2
# from websocket_server import WebsocketServer

# class FrameReader(Thread):
#     """
#     doc

#     Args:
#         Thread (_type_): _description_
#     """
#     def __init__(self, src, frame_queue: queue.Queue, killer, name = "CameraReader"):
#         """

#         Args:
#             src (_type_): _description_
#             frame_queue (queue.Queue): _description_
#             killer (_type_): _description_
#             name (str, optional): _description_. Defaults to "CameraReader".
#         """
#         super().__init__(daemon=True, name=name)
#         self.src = src
#         self.frame_queue = frame_queue
#         self.killer = killer

    
#     def put_latest_frame(self, frame):
#         """
#         Capture image function and write it on frame_queue for processing 

#         Args:
#             frame (np.ndarray): _description_
#         """
#         try:
#             self.frame_queue.put_nowait(frame)
#         except queue.Full:
#             try:
#                 self.frame_queue.get_nowait()
#             except queue.Empty:
#                 pass
#             self.frame_queue.put_nowait(frame)

#     def create_rtsp_gst_pipeline(self, rtsp_src):
#         return f"""
#                 rtspsrc location={rtsp_src} latency=0 drop-on-latency=true !
#                 rtph265depay ! h265parse ! mppvideodec fast-mode=true !
#                 queue max-size-buffers=1 leaky=downstream !
#                 rgaconvert ! video/x-raw, format=BGR, width=4000, height=3000 ! appsink drop=true max-buffers=1 sync=0
#             """

#     def create_vid_gst_pipeline(self, vid_src):
#         return f"""
#                 filesrc location={vid_src} ! qtdemux ! h265parse ! mppvideodec ! rgaconvert ! video/x-raw, format=BGR, width=4000, height=3000 !
#                 appsink drop=true max-buffers=1 sync=false
#             """

#     def run(self):
#         """
#         run FrameReader function to capture frames
#         """
#         try:
#             state, input_type = self.check_source_type(self.src)
#         except TypeError:
#             print("Input source is INVALID")
#             return
        
#         if not state:
#             print("Input source is INVALID")
#             assert TypeError
#             return
        
#         if input_type == "cam":
#             gst_pipeline = self.create_rtsp_gst_pipeline(self.src)
#         else:
#             gst_pipeline = self.create_vid_gst_pipeline(self.src)

#         cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

#         if not cap.isOpened():
#             print(f"[{self.name}] ERROR: cannot open source {self.src}")
#             self.frame_queue.put(object())
#             return

#         while not self.killer.is_stopped():
#             # for i in range(2):
#             #     cap.grab()
#             # ret, frame = cap.retrieve()
#             ret, frame = cap.read()
#             if not ret:
#                 print(f"[{self.name}] End of stream.")
#                 break
            
#             self.put_latest_frame(frame)

#             # time.sleep(0.002)
#         cap.release()
#         if self.frame_queue.full():
#             self.frame_queue.get_nowait()
#         else:
#             self.frame_queue.put(object())
#         self.killer.stop()
#         print(f"[{self.name}] Stopped.")

#     def check_source_type(self, src):
#         if src.startswith("rtsp://"):
#             return True, "cam"
#         elif src.lower().endswith(("mp4", "avi", "mov", "mkv")):
#             return True, "vid"
#         else:
#             return False, None

#     # def check_rtsp_valid(self,):
#     #     pass

# class VideoWriter(Thread):
#     """Thread that writes frames to a video file."""
#     def __init__(self, processed_queue: queue.Queue, output_path: str, fps: float, killer, name="VideoWriter"):
#         super().__init__(daemon=True, name=name)
#         self.processed_queue = processed_queue
#         self.output_path = output_path
#         self.fps = fps
#         self.killer = killer
#         self.server = WebsocketServer("127.0.0.1", port=9000)

#     def create_gst_write_pipeline(self, fname):
#         return f"""
#                 appsrc ! video/x-raw, format=BGR, width=1600, height=1200 !
#                 rgaconvert ! video/x-raw, format=NV12, width=1600, height=1200 !
#                 mpph265enc rc-mode=vbr bps=4000000 gop=60 !
#                 h265parse ! mp4mux ! filesink location={fname} sync=false
#             """

#     def run(self):
#         first_frame = None
#         ws_thread = Thread(target=self.server.run_forever, daemon=True)
#         ws_thread.start()
#         while not self.killer.is_stopped():
#             processed_frame = self.processed_queue.get()
#             if processed_frame is object():
#                 print(f"[{self.name}] No frames to write. Exiting.")
#                 return
#             first_frame = processed_frame
#             break

#         h, w = first_frame.shape[:2]
#         out = cv2.VideoWriter(self.create_gst_write_pipeline(self.output_path), cv2.CAP_GSTREAMER, 0, self.fps, (w,h), True)
#         if not out.isOpened():
#             print(f"[{self.name}] ERROR: cannot open output {self.output_path}")
#             return


#         out.write(first_frame)

#         # print(f"[{self.name}] Writing to {self.output_path} at {self.fps} FPS")

#         while not self.killer.is_stopped():
#             processed_frame = self.processed_queue.get()
#             if processed_frame is object():
#                 break
#             # cv2.imwrite("/home/pi/car-detector/public/camera/front_img.jpg", processed_frame)
#             out.write(processed_frame)
#             # t1 = time.perf_counter()
#             self.send_frame(processed_frame)
#             # print(f"time send image:{time.perf_counter() - t1}")

#             # time.sleep(0.002)

#         out.release()
#         self.server.shutdown()
#         ws_thread.join()
#         print(f"[{self.name}] Stopped.")
    
#     def send_frame(self, frame):
#         ui_frame = cv2.resize(frame, None, fx=0.25, fy=0.25)
#         _, ui_frame_encode = cv2.imencode(".jpg", ui_frame)#, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
#         b64_frame = base64.b64encode(ui_frame_encode).decode("utf-8")
#         self.server.send_message_to_all(b64_frame)

    # def send_to_RTSP(self, processed_queue, rtsp_url):
    #     width = 1280
    #     height = 720
    #     fps = 5
    #     ffmpeg_cmd = [
    #         "ffmpeg",
    #         "-y",  # Overwrite output files
    #         "-f", "rawvideo",  # Input format
    #         "-pixel_format", "bgr24",  # Input pixel format
    #         "-video_size", f"{width}x{height}",  # Input resolution
    #         "-framerate", str(fps),  # Input frame rate
    #         "-i", "-",  # Input from stdin
    #         "-c:v", "libx264",  # Encoder
    #         "-preset", "ultrafast", 
    #         "-g", "10", # Encoding speed
    #         "-f", "rtsp",  # Output format
    #         "-tune", "zerolatency",
    #         rtsp_url,  # RTSP output URL
    #     ]
    #     process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    #     return process        



# class TestWrite:
#     def test_write(self, frame_queue, processed_queue, stop_event):
#         while not stop_event.is_set():
#             time.sleep(0.05)
#             processed_queue.put(frame_queue.get())


import time
from threading import Thread, Lock

import numpy as np
import gi
gi.require_version("Gst", "1.0")
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GLib, GstApp

Gst.init(None)


class Camera:
    """
    Hardware-accelerated RTSP H.265 capturing for using NanoPi R6S board
    This class is using Gstreamer pipeline, MPP and RGA as plugin for reading frames
    """
    def __init__(
        self,
        rtsp_url: str,
        width: int = 2560,
        height: int = 1440,
        cam_id: int = 0,
        use_tcp: bool = True,
    ):
        # setup RTSP camera settings
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.cam_id = cam_id
        self.use_tcp = use_tcp

        # FPS tracking
        self._fps = 0
        self._frame_counter = 0
        self._fps_start_time = time.perf_counter()

        # setup frame states
        self._lock = Lock()
        self._latest_frame = None
        self._frame_id = 0
        self._dropped_count = 0
        self._last_pull_time = 0

        # Build pipeline
        self.pipeline = self._build_pipeline()
        self.appsink = self.pipeline.get_by_name("sink")

        # Gstreamer bus for monitoring
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        # Reader thread
        self._running = False
        self._reader_thread = None
    
    def _build_pipeline(self) -> Gst.Pipeline:
        """
        rtspsrc (low latency)
        """
        
        protocols = "tcp" if self.use_tcp else "udp"
        
        pipeline_str = (
            f" rtspsrc location={self.rtsp_url} latency={0} drop-on-latency=true !"
            f" rtph265depay !"
            f" h265parse !"
            f" mppvideodec fast-mode=true !"  # Critical: fast-mode skips post-processing
            f" queue max-size-buffers=1 leaky=2 !"
            f" rgaconvert !"
            f" video/x-raw, format=RGB, width={self.width}, height={self.height} !"
            f" appsink name=sink max-buffers=1 drop=true sync=false"
            #f"emit-signals=false "
        )
        
        try:
            pipeline = Gst.parse_launch(pipeline_str)
            return pipeline
        except GLib.Error as e:
            raise RuntimeError(f"Failed to create RTSP pipeline for {self.rtsp_url}: {e}")

    def _on_bus_message(self, message):
        """Handle pipeline errors and warnings."""
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[Cam{self.cam_id}] Pipeline error: {err}, {debug}")
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"[Cam{self.cam_id}] Warning: {warn}")
        elif t == Gst.MessageType.EOS:
            print(f"[Cam{self.cam_id}]")

    def _reader_loop(self):
        """Background thread that pulls frames from appsink gstreamer."""
        while self._running:
            sample = self.appsink.try_pull_sample(Gst.SECOND)
            
            if sample is None:
                continue
            buffer = sample.get_buffer()
            # caps = sample.get_caps()
            
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                continue
            
            try:
                # RGB format: height × width × 3
                frame = np.ndarray(
                    shape=(self.height, self.width, 3),
                    dtype=np.uint8,
                    buffer=map_info.data
                )
                frame_copy = frame.copy()

            finally:
                buffer.unmap(map_info)

            with self._lock:
                self._latest_frame = frame_copy
                self._frame_id += 1

            # Update FPS counter
            self._frame_counter += 1
            # now = time.perf_counter()
            # elapsed = now - self._f
            

            # if elapsed >= 1.0:
            #     self._fps = self._frame_counter / elapsed
            #     self._frame_counter = 0
            #     self._fps_start_time = now

        print(f"[Cam{self.cam_id}] Reader thread stopped")

    def start(self):
        """Start the camera capture pipeline."""
        if self._running:
            print(f"[Cam{self.cam_id}] Already running")
            return

        print(f"[Cam{self.cam_id}] Starting camera...")

        # Build pipeline
        self.pipeline = self._build_pipeline()
        self.appsink = self.pipeline.get_by_name("sink")

        if self.appsink is None:
            raise RuntimeError(f"[Cam{self.cam_id}] Failed to get appsink element")

        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self._on_bus_message)

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"[Cam{self.cam_id}] Failed to start pipeline")

        self._running = True
        self._reader_thread = Thread(target=self._reader_loop,
                                     daemon=True,
                                     name=f"Cam{self.cam_id}_Reader")
        self._reader_thread.start()

        print(f"[Cam{self.cam_id}] Started successfully")

    def stop(self):
        """Stop the camera capture pipeline."""
        if not self._running:
            return

        print(f"[Cam{self.cam_id}] Stopping camera...")

        self._running = False

        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=3)
            if self._reader_thread.is_alive():
                print(f"[Cam{self.cam_id}] Warning: Reader thread did not stop cleanly")

        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

        if self.bus:
            self.bus.remove_signal_watch()

        with self._lock:
            self._latest_frame = None
            self._frame_id = 0
            self._is_ready = False

        print(f"[Cam{self.cam_id}] Stopped")

    def get_frame(self):
        """
        Get the latest frame base on thread safe method.
        
        Returns:
            tuple: (frame_id, frame_array) or (None, None) if no frame available
        """
        with self._lock:
            if self._latest_frame is None:
                return None, None
            return self._frame_id, self._latest_frame.copy()

    def get_fps(self):
        """
        Get current capture FPS.
        
        Returns:
            float: Current FPS
        """
        return self._fps

    def is_ready(self):
        """
        Check if camera is ready and producing frames.
        
        Returns:
            bool: True if ready, False otherwise
        """
        with self._lock:
            return self.is_ready and self._latest_frame is not None

    def get_info(self):
        """
        Get camera information.
        
        Returns:
            dict: Camera info including resolution, FPS, frame count
        """
        with self._lock:
            return {
                "cam_id": self.cam_id,
                "rtsp_url": self.rtsp_url,
                "resolution": (self.width, self.height),
                "fps": self._fps,
                "frame_id": self._frame_id,
                "is_ready": self._is_ready and self._latest_frame is not None
            }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

'''
****----    sample using Camera class:
cam = RTSPCamera("rtsp://192.168.1.100:554/stream", cam_id=0)
cam.start()

# Wait for camera to be ready
while not cam.is_ready():
    time.sleep(0.1)

# Use in pipeline
while True:
    frame_id, frame = cam.get_frame()
    if frame is not None:
        # Process frame
        pass
cam.stop()


with RTSPCamera("rtsp://192.168.1.100:554/stream", cam_id=0) as cam:
    while not cam.is_ready():
        time.sleep(0.1)
    
    while True:
        frame_id, frame = cam.get_frame()
        if frame is not None:

'''
