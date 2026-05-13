import os
import sys
sys.path.insert(0, str(os.getcwd()))

from src.detection.inference import InferenceDetRKNN

import cv2
import numpy as np
_LABELS = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5',
    6: '6', 7: '7', 8: '8', 9: '9', 10: 'alef',
    11: 'b', 12: 'je', 13: 'lam', 14: 'mim', 15: 'noon',
    16: 'qaf', 17: 'vav', 18: 'h', 19: 'ye', 20: 'dal',
    21: 'sin', 22: 'sad', 23: 'malol', 24: 'te',
    25: 'ta', 26: 'ein', 27: 'diplomat', 28: 'siyasi',
    29: 'p', 31: 'the', 32: 'ze', 33: 'shin', 34: 'fe',
    35: 'kaf', 36: 'gaf'
}

_SUB_CHAR_LABEL = {17, 11, 26, 34, 24, 21, 28, 32, 16,
                    22, 15, 23, 35, 18, 10, 20, 33, 25,
                    27, 29, 19, 13, 14, 31, 12, 36}

_SUB_CHAR_LABEL_NUM = set(range(10))
def remove_dup(boxes, clss, scores, x_pixel_thresh=3, expected_chars=8):
    """
    Docstring for remove_dup
    
    :param det: Description
    :param x_pixel: Description
    :param expect_char: Description
    """
    n = len(boxes)
    # if n <= 8:
    #     return boxes, clss, scores
    i = 0
    keep_indc = []
    while i < n:
        j = i + 1
        while j<n and ((boxes[j,2] + boxes[j,0]) - (boxes[i, 2] + boxes[i, 0]))/2 <= x_pixel_thresh:
            j += 1
        if j - i > 1:
            best_local_idx = np.argmax(scores[i:j])
            keep_indc.append(i + best_local_idx)
        else:
            keep_indc.append(i)
        i = j
    filtered_boxes = boxes[keep_indc]
    filtered_scores = scores[keep_indc]
    filtered_classes = clss[keep_indc]
    if len(filtered_boxes) > expected_chars:
        top_n_indc = np.argpartition(filtered_scores, -expected_chars)[-expected_chars:]
        top_n_indc = top_n_indc[np.argsort(filtered_boxes[top_n_indc, 0])]

        filtered_boxes = filtered_boxes[top_n_indc]
        filtered_scores = filtered_scores[top_n_indc]
        filtered_classes = filtered_classes[top_n_indc]

    return filtered_boxes, filtered_classes, filtered_scores, len(filtered_boxes)

def format_license_plate(chars) -> str:
    return f"{chars[0]}{chars[1]}_{chars[2]}_{chars[3]}{chars[4]}{chars[5]}_{chars[6]}{chars[7]}"

def lpr_process():
    model = InferenceDetRKNN(
        model_path="/home/pi/NanoPi-R6S-ALPCMR/src/models/LPR_160_v4.1.0.rknn",
        img_size=(160,160),
        model_branch=2,
        obj_thresh=0.4,
        nms_thresh=0.6,
        pre_nms_topk=100,
        max_det=8,
        keep_multi_class=False,
        reg_max=1,
        npu_core=1,
        use_dfl=False
    )
    img = cv2.imread("lp.jpg")
    boxes, clss, scores  = model.end_to_end_inference(img)
    order = boxes[:,0].argsort()
    lp_data = remove_dup(boxes[order], clss[order], scores[order])
    print(lp_data)
    if (                    
        lp_data[3] == 8 and 
        all(lp_data[1][j] in _SUB_CHAR_LABEL_NUM for j in (0, 1, 3, 4, 5, 6, 7)) and
        lp_data[1][2] in _SUB_CHAR_LABEL
    ):
        prob = round(sum(lp_data[2])/8, 4)
        print(prob)
        print(lp_data[1])
        lp_text = [_LABELS[int(lp_data[1][i])] for i in range(8)]
        format_license_plate(lp_text)


lpr_process()