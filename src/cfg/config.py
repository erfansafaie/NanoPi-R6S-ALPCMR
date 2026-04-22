



vDetModel = "/home/pi/lp/lpMain/models/vd_640_v11_rknn_model"

lpDModel = "/home/pi/lp/lpMain/models/LPD_v4.0.0_rknn_model"

lpRModel = "/home/pi/lp/lpMain/models/LPR_160_v4.1.0_rknn_model"

MODEL_PATH_LIST = [vDetModel, lpDModel, lpRModel]



# ipCamAddress = "rtsp://admin:tlJwpbo6@123@192.168.1.225"
# ipCamAddress = "rtsp://admin:Admin@123@192.168.59.226"
IP_CAM_ADDRESS = "rtsp://192.168.1.124:554/live/av0"
# Hint example: ipCamAddress = "rtsp://<username>:<password>@192.168.1.64/1"

RTSPURL = "rtsp://127.0.0.1:8554/live"

MAX_KEEP_FRAME = 90
PROCESS_FRAME_RATE = 10

CONFIG_DB_PATH = "/home/pi/car-detector/database/config.db"
MAIN_DB_PATH = "/home/pi/car-detector/database/data.db"