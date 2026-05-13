import time
import os
import sys
sys.path.insert(0,str(os.getcwd()))

from src.camera.read_cam import Camera








cam = Camera("rtsp://192.168.168.52:554/stream1", cam_id=0)
cam.start()

# Wait for camera to be ready
while not cam.is_ready():
    time.sleep(0.1)

# Use in pipeline
while True:
    t1 = time.perf_counter()
    frame_id, frame = cam.get_frame()
    print(time.perf_counter() - t1)
    # if frame is not None:
    #     # Process frame
    #     pass
cam.stop()