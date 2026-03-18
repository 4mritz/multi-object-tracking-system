import cv2
from core.tracking import tracker
from core.tracking.tracker import MultiObjectTracker
from models.detection.yolo import YOLODetector

def run(video_path):
    cap = cv2.VideoCapture(video_path)

    tracker = MultiObjectTracker()
    detector = YOLODetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        tracker.predict()
        tracker.update(frame, detections)  

        for track in tracker.get_active_tracks():
            x1, y1, x2, y2 = map(int, track.bbox)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID {track.id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run("data/raw/sydney_walking.mp4")