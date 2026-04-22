from ultralytics import YOLO




model = YOLO("export/ptModel/vDet640.pt", task="detect")

model.export(format="ncnn", imgsz=(192,320), half=True)

# model = YOLO("/home/pi/licensePlate/lpMain/export/ptModel/LPDv3.pt", task="detect")
# model.predict("lpMain/116.jpg", imgsz=(384,640), conf=0.5)

