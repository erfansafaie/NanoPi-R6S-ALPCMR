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
import base64
# import websockets
# import asyncio
from websocket_server import WebsocketServer
from threading import Thread, Lock, Event
from queue import Queue, Full, Empty
from typing import Any, Optional, Union
from dataclasses import dataclass


import cv2
import numpy as np
import gi
gi.require_version("Gst", "1.0")
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GLib, GstApp

Gst.init(None)


class Camera:
    """
    Hardware-accelerated RTSP H.265 capture for NanoPi R6S.

    Produces FramePacket objects into a latest-frame queue:

        FramePacket(
            camera_id,
            frame_id,
            timestamp,
            image
        )

    This class is designed for the threaded pipeline:

        Camera -> frame_queue -> detect+track worker
    """

    def __init__(
        self,
        frame_queue: Queue,
        stream_queue: Queue,
        src_address: str,
        killer: Any,
        cam_id: Union[int, str],
        width: int = 1920,
        height: int = 1080,
        use_tcp: bool = False,
        queue_drop_old: bool = True,
        pull_timeout_sec: float = 0.1,
    ):
        
        self.frame_queue = frame_queue
        self.stream_queue = stream_queue
        self.src_address = src_address
        self.killer = killer
        self.cam_id = cam_id

        self.width = width
        self.height = height
        self.use_tcp = use_tcp
        self.queue_drop_old = queue_drop_old
        self.pull_timeout_sec = pull_timeout_sec

        self.pipeline: Optional[Gst.Pipeline] = None
        self.appsink = None
        self.bus = None

        self._reader_thread: Optional[Thread] = None
        self._running = False

        self._ready = False

        self._stats_lock = Lock()

        self._build_pipeline()

    def check_source_type(self, src):
        if src.startswith("rtsp://"):
            return True, "cam"
        elif src.lower().endswith(("mp4", "avi", "mov", "mkv")):
            return True, "vid"
        else:
            return False, None

    def _build_pipeline(self):
        _,  src = self.check_source_type(self.src_address)
        if src == "cam":
            self.pipeline = self._build_cam_pipeline()
        elif src == "vid":
            self.pipeline = self._build_vid_pipeline()
            print(f"[{src}] : source")
        else:
            raise ValueError(f"[Cam{self.cam_id}] Unknown source type: {self.src}")

        self.appsink = self.pipeline.get_by_name("sink")
        if self.appsink is None:
            raise RuntimeError(f"[Cam{self.cam_id}] Failed to get appsink element")

        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self._on_bus_message)

    def _build_cam_pipeline(self) -> Gst.Pipeline:
        """
        Low-latency RTSP H.265 pipeline using Rockchip MPP + RGA.
        """

        protocols = "tcp" if self.use_tcp else "udp"

        pipeline_str = (
            f"rtspsrc location={self.src_address} "
            f"latency=0 "
            f"drop-on-latency=true "
            f"protocols={protocols} ! "
            f"rtph265depay ! "
            f"h265parse ! "
            f"mppvideodec fast-mode=true ! "
            f"queue max-size-buffers=1 leaky=2 ! "
            f"rgaconvert ! "
            f"video/x-raw,format=RGB,width={self.width},height={self.height} ! "
            f"appsink name=sink max-buffers=1 drop=true sync=false emit-signals=false"
        )

        try:
            return Gst.parse_launch(pipeline_str)
        except GLib.Error as e:
            raise RuntimeError(
                f"[Cam{self.cam_id}] Failed to create RTSP pipeline: {e}"
            )

    def _build_vid_pipeline(self) -> Gst.Pipeline:
        """
        Local H.265 video file pipeline.
        """

        pipeline_str = (
            f"filesrc location={self.src_address} ! "
            f"qtdemux ! "
            f"h265parse ! "
            f"mppvideodec fast-mode=true ! "
            #f"queue max-size-buffers=1 leaky=2 ! "
            f"rgaconvert ! "
            f"video/x-raw,format=RGB,width={self.width},height={self.height} ! "
            f"appsink name=sink max-buffers=30 drop=false sync=true emit-signals=false"
        )

        try:
            return Gst.parse_launch(pipeline_str)
        except GLib.Error as e:
            raise RuntimeError(
                f"[Cam{self.cam_id}] Failed to create video pipeline: {e}"
            )

    def _on_bus_message(self, message):
        """
        Handle pipeline messages.
        """
        msg_type = message.type

        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[Cam{self.cam_id}] Pipeline error: {err}, debug={debug}")

        elif msg_type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"[Cam{self.cam_id}] Pipeline warning: {warn}, debug={debug}")

        elif msg_type == Gst.MessageType.EOS:
            print(f"[Cam{self.cam_id}] End of stream")

        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending = message.parse_state_changed()
                print(
                    f"[Cam{self.cam_id}] State changed: "
                    f"{old_state.value_nick} -> {new_state.value_nick}"
                )

    def _push_latest(self, frame_pack: np.ndarray, frame):
        """
        Push newest frame to bounded queue.

        For real-time camera processing, if the queue is full,
        discard the old frame and insert the new one.
        """

        if self.frame_queue is None:
            return

        if self.queue_drop_old:
            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                    self.stream_queue.get_nowait()
            except Empty:
                pass

            try:
                self.frame_queue.put_nowait(frame_pack)
                self.stream_queue.put_nowait(frame)
            except Full:
                pass

        else:
            try:
                self.frame_queue.put_nowait(frame_pack)
                self.stream_queue.put_nowait(frame)
            except Full:
                pass


    def _reader_loop(self):
        """
        Pull frames from appsink and send FramePacket to pipeline queue.
        """

        print(f"[Cam{self.cam_id}] Reader thread started")
        frame_id = 0
        gst_timeout_ns = int(self.pull_timeout_sec * Gst.SECOND)
        while not self.killer.is_stopped() and self._running:

            self._ready = False
            try:
                sample = self.appsink.try_pull_sample(gst_timeout_ns)
            except Exception as e:
                print(f"[Cam {self.cam_id}] appsink pull error: {e}")
                time.sleep(0.001)
                continue

            if sample is None:
                continue

            buffer = sample.get_buffer()
            if buffer is None:
                continue

            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                continue

            try:
                frame = np.ndarray(
                    shape=(self.height, self.width, 3),
                    dtype=np.uint8,
                    buffer=map_info.data,
                ).copy()

            finally:
                buffer.unmap(map_info)
            t_now = time.time()
            frame_id += 1
            frame_pack = (frame, self.cam_id, frame_id, t_now)
            self._push_latest(frame_pack, frame)
            self._ready = True

            time.sleep(0.02)


        print(f"[Cam {self.cam_id}] Reader thread stopped")

    def start(self):
        """
        Start camera pipeline and reader thread.
        """

        if self._running:
            print(f"[Cam {self.cam_id}] Already running")
            return

        print(f"[Cam {self.cam_id}] Starting camera...")

        if self.pipeline is None:
            self._build_pipeline()

        ret = self.pipeline.set_state(Gst.State.PLAYING)

        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"[Cam {self.cam_id}] Failed to start pipeline")

        self._running = True

        self._reader_thread = Thread(
            target=self._reader_loop,
            daemon=True,
            name=f"Cam {self.cam_id}_Reader",
        )
        self._reader_thread.start()

        print(f"[Cam {self.cam_id}] Started successfully")

    def stop(self):
        """
        Stop camera reader and GStreamer pipeline.
        """

        if not self._running:
            return

        print(f"[Cam {self.cam_id}] Stopping camera...")

        self._running = False

        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=3.0)
            if self._reader_thread.is_alive():
                print(f"[Cam {self.cam_id}] Warning: reader thread did not stop cleanly")

        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)

        if self.bus is not None:
            try:
                self.bus.remove_signal_watch()
            except Exception:
                pass

        with self._stats_lock:
            self._ready = False

        print(f"[Cam {self.cam_id}] Stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


class StreamWS:
    def __init__(
            self,
            killer,
            websocket_url: str,
            websocket_port: int,
            stream_queue: Queue,
    ):
        self.websocket_url = websocket_url
        self.stream_queue = stream_queue
        self.killer = killer

        self.server = WebsocketServer(websocket_url, port=websocket_port)
        self.ws_thread = Thread(target=self.server.run_forever, daemon=True)
        self.ws_thread.start()

    def _resize_encode(self, image):
        """
        Resize and Encode to jpeg input image. Prepare image array to send base64 on web-scoket
        """
#         _, ui_frame_encode = cv2.imencode(".jpg", ui_frame)#, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
#         b64_frame = base64.b64encode(ui_frame_encode).decode("utf-8")
        return base64.b64encode(
            cv2.imencode(
            ".jpg",
            cv2.cvtColor(
            cv2.resize(image, None, fx=0.33, fy=0.33, interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 50])[1]
        ).decode("utf-8")

    def run(self):
        send_thread = Thread(target=self._send_loop, daemon=True)
        send_thread.start()

    def _send_loop(self):
        while not self.killer.is_stopped():
            try:
                frame = self.stream_queue.get(timeout=0.05)
            except Empty:
                continue
            b64_frame = self._resize_encode(frame)
            self.server.send_message_to_all(b64_frame)
            time.sleep(0.033)
        self.server.shutdown()
        self.ws_thread.join()
# class SteramWS:
#     def __init__(
#             self,
#             killer,
#             websocket_url: str,
#             stream_queue: Queue,
#             sleep_idle: float = 0.005,

#     ):
#         self.websocket_url = websocket_url
#         self.stream_queue = stream_queue
#         self.killer = killer

#         self.sleep_idle = sleep_idle


#     def _resize_encode(self, image):
#         """
#         Resize and Encode to jpeg input image. Prepare image array to send base64 on web-scoket
#         """
# #         _, ui_frame_encode = cv2.imencode(".jpg", ui_frame)#, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
# #         b64_frame = base64.b64encode(ui_frame_encode).decode("utf-8")
#         return base64.b64encode(
#             cv2.imencode(
#             ".jpg",
#             cv2.resize(image, fx=0.33, fy=0.33, interpolation=cv2.INTER_LINEAR),
#             [int(cv2.IMWRITE_JPEG_QUALITY), 75])[1]
#         ).decode("utf-8")

#     async def _send_loop(self, ws):
#         while not self.killer.is_stopped():
#             frame = self.stream_queue.get_nowait()
#             if frame is None:
#                 await asyncio.sleep(self.idle_sleep)
#                 continue
#             try:
#                 b64_str = await asyncio.to_thread(
#                     self.resize_encode,
#                     frame
#                 )
                
#                 await asyncio.wait_for(
#                     ws.send(b64_str),
#                     timeout=2
#                 )
#             except asyncio.TimeoutError:
#                 break


#     async def send_stream(self):
#         while not self.killer.is_stopped():
#             try:
#                 async with websockets.connect(
#                     self.websocket_url,

#                 ) as ws:
#                     await self._send_loop(ws)
#             except asyncio.CancelledError:
#                 print(f"[WS] Cancelled")
#                 raise

#             except Exception as e:
#                 print(f"[WS] Connection error: {e}")
#                 if self.killer.is_stopped():
#                     break

#                 await asyncio.sleep(2)#self.reconnect_delay)
#         print(f"[WS] Stopped")


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
