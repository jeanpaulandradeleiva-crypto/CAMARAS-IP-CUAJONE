from ultralytics import YOLO

model = YOLO("yolo11n.pt")

url = "rtsp://admin:Omate$01@172.19.90.72/axis-media/media.amp"

model.predict(
    source=url,
    show=True,
    classes=[0, 2, 7],
    conf=0.4
)