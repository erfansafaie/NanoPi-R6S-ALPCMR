import os
import sys
sys.path.insert(0, str(os.getcwd()))
import time

import cv2

from src.detection.inference import InferenceClsRKNN

COLOR_LABELS = ('Black', 'Blue', 'Brown', 'Dark Green', 'Dark Red', 'Gold',
            'Gray', 'Green', 'Orange', 'Red', 'Silver', 'White', 'Yellow')
MODEL_LABELS = ('Arisun', 'Atlas', 'Bahman Fidelity', 'Baic Sabrina',
                'Chery Tiggo 5', 'Dena', 'Fownix FX', 'H30Cross', 'Haima S7', 'Jac J4',
                'Jac J5', 'Jac S3', 'Jac S5', 'KMC T8', 'KaraMazdaPickup', 'Kia Cerato',
                'Kia Mohave', 'MVM315H', 'MVMX22', 'NeissanVanet', 'Peugeot_206',
                'Peugeot_206_SD', 'Peugeot_207', 'Peugeot_405', 'Peugeot_Pars',
                'PeykanSavari', 'PeykanVanet', 'Pride_Nasim', 'Pride111', 'Pride131',
                'Pride132', 'Pride141', 'Pride151', 'Quik', 'Renault_L90', 'Renault_Sandro',
                'RenaultPK', 'RioSD', 'Runna', 'Saina', 'Samand', 'SamandSoren', 'Shahin',
                'Tara', 'Tiba', 'Tiba2', 'Xantia')

model_color = InferenceClsRKNN(
    "/home/pi/NanoPi-R6S-ALPCMR/src/models/car_color_classification_128_13cls_v3.rknn",
    img_size=(128,128),
    npu_core=2
)
model_type = InferenceClsRKNN(
    "/home/pi/NanoPi-R6S-ALPCMR/src/models/car_model_classification_128_47cls_v4.rknn",
    img_size=(128,128),
    npu_core=2
)

img = cv2.imread("v.jpg")
t1 = time.perf_counter()
img = model_color.preprocess(img, target_size=(128,128))
out_color, prob_color = model_color.run(img)
out_model, prob_model = model_type.run(img)
print(time.perf_counter() - t1)
print(COLOR_LABELS[out_color], prob_color)
print(MODEL_LABELS[out_model], prob_model)
