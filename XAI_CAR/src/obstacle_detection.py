import cv2
from ultralytics import YOLO


class ObstacleDetector:
    """
    Obstacle detection using YOLOv8 (Ultralytics).

    Returns:
    {
        'frame': frame_with_boxes,
        'detections': [
            {
                'xyxy': [x1, y1, x2, y2],
                'conf': confidence,
                'class': class_id,
                'label': class_name
            }
        ]
    }
    """

    def __init__(self, weights_path, device='cpu', conf_thres=0.4):
        self.device = device
        self.conf_thres = conf_thres
        self.model = None

        try:
            self.model = YOLO(weights_path)
            self.model.conf = conf_thres  # global confidence threshold
            print("[INFO] YOLOv8 loaded successfully")

        except Exception as e:
            print("[WARN] YOLOv8 not available, obstacle detection disabled")
            print("[WARN]", e)
            self.model = None

    def detect(self, frame):
        overlay = frame.copy()
        detections = []

        if self.model is None:
            return {
                'frame': overlay,
                'detections': detections
            }

        # YOLOv8 inference (BGR frame is fine)
        results = self.model(
            frame,
            device=self.device,
            verbose=False
        )[0]

        if results.boxes is None:
            return {
                'frame': overlay,
                'detections': detections
            }

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_thres:
                continue

            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )
            cv2.putText(
                overlay,
                f"{label} {conf:.2f}",
                (x1, max(y1 - 7, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

            detections.append({
                'xyxy': [float(x1), float(y1), float(x2), float(y2)],
                'conf': conf,
                'class': cls_id,
                'label': label
            })

        return {
            'frame': overlay,
            'detections': detections
        }


# ----------------------------
# Standalone webcam test
# ----------------------------
if __name__ == "__main__":
    od = ObstacleDetector("models/yolov8n.pt", device="cpu")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = od.detect(frame)
        cv2.imshow("YOLOv8 Obstacle Detection", res["frame"])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
