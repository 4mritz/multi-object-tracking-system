from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.4):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)[0]

        detections = []

        if results.boxes is None:
            return detections

        for box in results.boxes.data:
            x1, y1, x2, y2, score, cls = box.tolist()

            detections.append([x1, y1, x2, y2])

        return detections