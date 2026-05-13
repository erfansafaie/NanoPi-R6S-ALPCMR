import os
import sys
sys.path.insert(0, str(os.getcwd()))


from src.detection.inference import InferenceDetRKNN

import cv2


def vehicle_process():
    model = InferenceDetRKNN(
    model_path="/home/pi/NanoPi-R6S-ALPCMR/src/models/vd_640_v11.rknn",
    img_size=(192,320),
    model_branch=3,
    obj_thresh=0.4,
    nms_thresh=0.6,
    pre_nms_topk=100,
    max_det=10,
    keep_multi_class=False,
    reg_max=16,
    npu_core=0,
    use_dfl=True
)
    img = cv2.imread("3.png")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    box, clss, scores  = model.end_to_end_inference(img)
    v =img[int(box[0][1]):int(box[0][3]), int(box[0][0]):int(box[0][2])]
    cv2.imwrite("v.jpg", v)

def lp_process():
    model = InferenceDetRKNN(
        model_path="/home/pi/NanoPi-R6S-ALPCMR/src/models/LPD_v4.0.0.rknn",
        img_size=(160,160),
        model_branch=1,
        obj_thresh=0.4,
        nms_thresh=0.6,
        pre_nms_topk=10,
        max_det=1,
        keep_multi_class=False,
        reg_max=1,
        npu_core=1,
        use_dfl=False
    )
    img = cv2.imread("v.jpg")
    box, *_  = model.end_to_end_inference(img)
    lp =img[int(box[0][1]):int(box[0][3]), int(box[0][0]):int(box[0][2])]
    cv2.imwrite("lp.jpg", lp)


lp_process()
# vehicle_process()