from ultralytics import YOLOv10

model = YOLOv10("yolov10x.pt")
model = YOLOv10("D:/university information/specialized course/pattern recognition/yolov10-main/yolov10-main/runs/detect/train13/weights/best.py")

model.export(format='onnx')