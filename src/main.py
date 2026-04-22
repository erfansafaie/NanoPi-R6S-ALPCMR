
"""
doc
"""
import sys
import argparse
import os
from queue import Queue
from pathlib import Path
from datetime import datetime

from camera.read_cam import FrameReader, VideoWriter
from run import RunWorker, GracefulKiller

sys.path.insert(0,"/home/pi/lp")
sys.path.insert(0,"/home/pi/lp/lpMain")
sys.path.insert(0,"/home/pi/lp/lpMain/detection")

# sys.path.insert(0,"/home/erfan/RKNNLP/lpMain")
# sys.path.insert(0,"/home/erfan/RKNNLP/lpMain/detection")



def main():
    """
    doc
    """


    parser = argparse.ArgumentParser(description="LP detection program")
    parser.add_argument("--source", required=True, default="A.mp4",#"rtsp://192.168.1.124:554/live/av0",
                        help="Specify source of processing. Camera, video or image path.")
    args = parser.parse_args()

    # source = "A.mp4"
    killer = GracefulKiller()
    frame_queue = Queue(maxsize=1)
    processed_queue = Queue(maxsize=10)

    frame_reader = FrameReader(args.source, frame_queue, killer)
    run_worker = RunWorker(frame_queue, processed_queue, killer)
    # out_vid_path = Path(args.source)
    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_writer = VideoWriter(processed_queue, dt+ '.mp4', 20, killer)

    frame_reader.start()
    run_worker.start()
    video_writer.start()
    input()
    # try:
    #     while not killer.is_stopped():
    #         time.sleep(0.001)
    #     # while not killer.is_stopped():
    #         # if not processed_queue.empty():
    #         #     frame = processed_queue.queue[-1]
    #         #     cv2.imshow("out", frame)
    #         # if cv2.waitKey(1) & 0xFF == ord('q'):
    #         #     killer.stop()
    #         #     break
    # except KeyboardInterrupt:
    #     print("\n[Main] KeyboardInterrupt. Stopping...")
    # finally:
    #     killer.stop()
    #     time.sleep(1)
    #     # frame_queue.put(object())
    #     # processed_queue.put(object())
    killer.stop()
    frame_reader.join()
    run_worker.join()
    video_writer.join()
    print("[Main] Done.")


if __name__ == "__main__":

    main()
