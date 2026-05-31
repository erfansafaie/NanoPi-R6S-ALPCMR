



v_det_model = "/home/pi/NanoPi-R6S-ALPCMR/src/models/vd_640_v11.rknn"
lpd_model = "/home/pi/NanoPi-R6S-ALPCMR/src/models/LPD_v4.0.0.rknn"
lpr_model = "/home/pi/NanoPi-R6S-ALPCMR/src/models/LPR_160_v4.1.0.rknn"
color_model = "/home/pi/NanoPi-R6S-ALPCMR/src/models/car_color_classification_128_13cls_v3.rknn"
model_model = "/home/pi/NanoPi-R6S-ALPCMR/src/models/car_model_classification_128_47cls_v4.rknn"

MODEL_PATH_LIST = [v_det_model, lpd_model, lpr_model, color_model, model_model]



# ipCamAddress = "rtsp://admin:tlJwpbo6@123@192.168.1.225"
# ipCamAddress = "rtsp://admin:Admin@123@192.168.59.226"
IP_CAM_ADDRESS = "rtsp://192.168.1.124:554/live/av0"
# Hint example: ipCamAddress = "rtsp://<username>:<password>@192.168.1.64/1"

RTSPURL = "rtsp://127.0.0.1:8554/live"

MAX_KEEP_FRAME = 90
PROCESS_FRAME_RATE = 10

CONFIG_DB_PATH = "/home/pi/car-detector/database/config.db"
MAIN_DB_PATH = "/home/pi/car-detector/database/data.db"