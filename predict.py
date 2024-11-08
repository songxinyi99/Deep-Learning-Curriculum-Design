from ultralytics import YOLOv10

model = YOLOv10("D:/university information/specialized course/pattern recognition/yolov10-main/yolov10-main/runs/detect/train13/weights/best.pt")
results = model.predict("D:/university information/specialized course/pattern recognition/yolov10-main/yolov10-main/data/valid/images/234.JPG")
results[0].show()