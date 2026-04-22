#
import subprocess
import queue
import time
import base64
from threading import Thread
import cv2
from websocket_server import WebsocketServer

class FrameReader(Thread):
    """
    doc

    Args:
        Thread (_type_): _description_
    """
    def __init__(self, src, frame_queue: queue.Queue, killer, name = "CameraReader"):
        """

        Args:
            src (_type_): _description_
            frame_queue (queue.Queue): _description_
            killer (_type_): _description_
            name (str, optional): _description_. Defaults to "CameraReader".
        """
        super().__init__(daemon=True, name=name)
        self.src = src
        self.frame_queue = frame_queue
        self.killer = killer

    
    def put_latest_frame(self, frame):
        """
        Capture image function and write it on frame_queue for processing 

        Args:
            frame (np.ndarray): _description_
        """
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            self.frame_queue.put_nowait(frame)

    def create_rtsp_gst_pipeline(self, rtsp_src):
        return f"""
                rtspsrc location={rtsp_src} latency=0 drop-on-latency=true !
                rtph265depay ! h265parse ! mppvideodec fast-mode=true !
                queue max-size-buffers=1 leaky=downstream !
                rgaconvert ! video/x-raw, format=BGR, width=4000, height=3000 ! appsink drop=true max-buffers=1 sync=0
            """

    def create_vid_gst_pipeline(self, vid_src):
        return f"""
                filesrc location={vid_src} ! qtdemux ! h265parse ! mppvideodec ! rgaconvert ! video/x-raw, format=BGR, width=4000, height=3000 !
                appsink drop=true max-buffers=1 sync=false
            """

    def run(self):
        """
        run FrameReader function to capture frames
        """
        try:
            state, input_type = self.check_source_type(self.src)
        except TypeError:
            print("Input source is INVALID")
            return
        
        if not state:
            print("Input source is INVALID")
            assert TypeError
            return
        
        if input_type == "cam":
            gst_pipeline = self.create_rtsp_gst_pipeline(self.src)
        else:
            gst_pipeline = self.create_vid_gst_pipeline(self.src)

        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            print(f"[{self.name}] ERROR: cannot open source {self.src}")
            self.frame_queue.put(object())
            return

        while not self.killer.is_stopped():
            # for i in range(2):
            #     cap.grab()
            # ret, frame = cap.retrieve()
            ret, frame = cap.read()
            if not ret:
                print(f"[{self.name}] End of stream.")
                break
            
            self.put_latest_frame(frame)

            # time.sleep(0.002)
        cap.release()
        if self.frame_queue.full():
            self.frame_queue.get_nowait()
        else:
            self.frame_queue.put(object())
        self.killer.stop()
        print(f"[{self.name}] Stopped.")

    def check_source_type(self, src):
        if src.startswith("rtsp://"):
            return True, "cam"
        elif src.lower().endswith(("mp4", "avi", "mov", "mkv")):
            return True, "vid"
        else:
            return False, None

    # def check_rtsp_valid(self,):
    #     pass

class VideoWriter(Thread):
    """Thread that writes frames to a video file."""
    def __init__(self, processed_queue: queue.Queue, output_path: str, fps: float, killer, name="VideoWriter"):
        super().__init__(daemon=True, name=name)
        self.processed_queue = processed_queue
        self.output_path = output_path
        self.fps = fps
        self.killer = killer
        self.server = WebsocketServer("127.0.0.1", port=9000)

    def create_gst_write_pipeline(self, fname):
        return f"""
                appsrc ! video/x-raw, format=BGR, width=1600, height=1200 !
                rgaconvert ! video/x-raw, format=NV12, width=1600, height=1200 !
                mpph265enc rc-mode=vbr bps=4000000 gop=60 !
                h265parse ! mp4mux ! filesink location={fname} sync=false
            """

    def run(self):
        first_frame = None
        ws_thread = Thread(target=self.server.run_forever, daemon=True)
        ws_thread.start()
        while not self.killer.is_stopped():
            processed_frame = self.processed_queue.get()
            if processed_frame is object():
                print(f"[{self.name}] No frames to write. Exiting.")
                return
            first_frame = processed_frame
            break

        h, w = first_frame.shape[:2]
        out = cv2.VideoWriter(self.create_gst_write_pipeline(self.output_path), cv2.CAP_GSTREAMER, 0, self.fps, (w,h), True)
        if not out.isOpened():
            print(f"[{self.name}] ERROR: cannot open output {self.output_path}")
            return


        out.write(first_frame)

        # print(f"[{self.name}] Writing to {self.output_path} at {self.fps} FPS")

        while not self.killer.is_stopped():
            processed_frame = self.processed_queue.get()
            if processed_frame is object():
                break
            # cv2.imwrite("/home/pi/car-detector/public/camera/front_img.jpg", processed_frame)
            out.write(processed_frame)
            # t1 = time.perf_counter()
            self.send_frame(processed_frame)
            # print(f"time send image:{time.perf_counter() - t1}")

            # time.sleep(0.002)

        out.release()
        self.server.shutdown()
        ws_thread.join()
        print(f"[{self.name}] Stopped.")
    
    def send_frame(self, frame):
        ui_frame = cv2.resize(frame, None, fx=0.25, fy=0.25)
        _, ui_frame_encode = cv2.imencode(".jpg", ui_frame)#, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        b64_frame = base64.b64encode(ui_frame_encode).decode("utf-8")
        self.server.send_message_to_all(b64_frame)

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