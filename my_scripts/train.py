from ultralytics import YOLO

model = YOLO("/root/yolov8n.pt")

results = model.train(
    data="/home/yolov8_data/mot_data.yaml",   # 远程主机文件路径
    epochs=200,
    imgsz=640,
    batch=16,
    device="0",
    project="runs/train",
    lr0=0.001
)