from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model("bus.jpg", save=True)
