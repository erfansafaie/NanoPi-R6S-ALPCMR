from ultralytics import YOLO
import time

import cv2


def test():
    cap = cv2.VideoCapture("3.mp4")
    model = YOLO("/home/pi/lp/lpMain/models/vd_640_v11_rknn_model", task="detect")

    while cap.isOpened():
        # for i in range(3):
        #     cap.grab()
        # ret, frame = cap.retrieve()
        ret, frame = cap.read()
        if ret:
            t1 = time.perf_counter()
            res = model.track(frame, conf=0.4, tracker="botsort.yaml",persist=True, stream=True)#,classes=[2,3,5,7])
            for r in res:
                print(time.perf_counter() - t1)

                f = r.plot()
                f = cv2.resize(f, (1920,1080))
                cv2.imshow("frame", f)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


    # model = YOLO("/home/pi/licensePlate/lpMain/models/single input/VD320_11_rknn_model", task="detect")
    # model.predict("frame_843.jpg", conf=0.4, imgsz=(256,320),save=True, save_crop=True)

    # model = YOLO("/home/pi/licensePlate/lpMain/models/lpd_160_340K_rknn_model", task="detect")
    # model.predict("f8.jpg", conf=0.25, half=True, imgsz=160, save_crop=True)
    # model.predict("f8.jpg", conf=0.25, half=True, imgsz=160, save_crop=True)
    # model.predict("f8.jpg", conf=0.25, half=True, imgsz=160, save_crop=True)
    # model.predict("f8.jpg", conf=0.25, half=True, imgsz=160, save_crop=True)
    # model.predict("f8.jpg", conf=0.25, half=True, imgsz=160, save_crop=True)

    # model = YOLO("/home/pi/licensePlate/lpMain/models/single input/LPR160_752K_rknn_model", task="detect")
    # res = model.predict("f8.jpg", conf=0.4, imgsz=160, save=True, save_crop=True)
    # for r in res:
    #     print(r.boxes.data)


test()