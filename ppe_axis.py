from ultralytics import YOLO

model = YOLO(r"C:\IA\PPE\PPE DETECTION.v14i.yolov8\runs\detect\train\weights\best.pt")

model.predict(
    source="rtsp://admin:Omate$01@172.19.90.72/axis-media/media.amp",
    show=True,
    conf=0.5
)