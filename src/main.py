
"""
doc
"""

import argparse
from queue import Queue
# from pathlib import Path
# from datetime import datetime
import sys
import os
sys.path.insert(0, str(os.getcwd()))
if str(os.getcwd()) != "/home/pi/NanoPi-R6S-ALPCMR":
    sys.path.insert(0, str(os.getcwd()) + "/NanoPi-R6S-ALPCMR")

from src.camera.read_cam import Camera, StreamWS
from src.run import RunWorker, GracefulKiller


# sys.path.insert(0,"/home/pi/lp")
# sys.path.insert(0,"/home/pi/lp/lpMain")
# sys.path.insert(0,"/home/pi/lp/lpMain/detection")

# sys.path.insert(0,"/home/erfan/RKNNLP/lpMain")
# sys.path.insert(0,"/home/erfan/RKNNLP/lpMain/detection")



def main():
    """
    doc
    """
    parser = argparse.ArgumentParser(description="LP detection program")
    parser.add_argument("--source1", required=True, default="A.mp4",#"rtsp://192.168.1.124:554/live/av0",
                        help="Specify source of processing. Camera, video or image path.")
    parser.add_argument("--source2", required=True, default="A.mp4",#"rtsp://192.168.1.124:554/live/av0",
                        help="Specify source of processing. Camera, video or image path.")
    args = parser.parse_args()

    # source = "A.mp4"
    killer = GracefulKiller()

    front_frame_queue = Queue(maxsize=10)
    rear_frame_queue = Queue(maxsize=10)
    stream_front_queue = Queue(maxsize=5)
    stream_rear_queue = Queue(maxsize=5)
    result_queue = Queue(maxsize=10)


    front_camera = Camera(
        front_frame_queue,
        stream_front_queue,
        args.source1,
        killer, 
        "F")
    rear_cam = Camera(
        rear_frame_queue,
        stream_rear_queue,
        args.source2,
        killer, 
        "R")

    front_stream = StreamWS(
        killer,
        "192.168.1.8",
        8765,
        stream_front_queue
    )
    rear_stream = StreamWS(
        killer,
        "192.168.1.8",
        8766,
        stream_rear_queue
    )
    run_worker = RunWorker(front_frame_queue, rear_frame_queue, result_queue, killer)

    # out_vid_path = Path(args.source)
    # dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # video_writer = VideoWriter(processed_queue, dt+ '.mp4', 20, killer)
    run_worker.run()
    front_camera.start()
    rear_cam.start()
    front_stream.run()
    rear_stream.run()
    input()
    front_camera.stop()
    rear_cam.stop()
    killer.stop()
    run_worker.stop()

    print("[Main] Done.")


if __name__ == "__main__":

    main()
