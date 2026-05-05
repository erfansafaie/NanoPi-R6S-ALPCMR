import numpy as np
from src.detection.inference import InferenceRKNN



def test_vehicle_detector():
    model_path = ""
    vehicle_model = InferenceRKNN()