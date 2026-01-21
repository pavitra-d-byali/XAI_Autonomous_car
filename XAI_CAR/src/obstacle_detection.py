import torch
import cv2
import numpy as np
from utils import draw_bbox


class ObstacleDetector:
    def __init__(self, weights_path=None, device='cpu', conf_thres=0.3):
        self.device = device
        self.conf_thres = conf_thres
        self.model = None

        # ----------------------------
        # Try loading YOLOv5 safely
        # ----------------------------
        try:
            if weights_path and weights_path.endswith('.pt'):
                # Check for empty / invalid weight file
                try:
                    with open(weights_path, 'rb') as f:
                        f.read(1)
                except Exception:
                    raise RuntimeError("YOLO weight file is missing or empty")

                self.model = torch.hub.load(
                    'ultralytics/yolov5',
                    'custom',
                    path=weights_path,
                    force_reload=False
                )
            else:
                # fallback to pretrained yolov5s
                self.model = torch.hub.load(
                    'ultralytics/yolov5',
                    'yolov5s',
                    pretrained=True
                )

            self.model.to(self.device)
            self.model.conf = conf_thres
            print("[INFO] YOLOv5 loaded successfully")

        except Exception as e:
            print("[WARN] YOLOv5 not available, running without obstacle detection")
            print("[WARN]", e)
            self.model = None

    def detect(self, frame):
        """
        Returns:
        {
            'frame': overlay frame,
            'detections': list of detections
        }
        """
        overlay = frame.copy()
        detections = []

        # ----------------------------
        # If YOLO not loaded → bypass
        # ----------------------------
        if self.model is None:
            return {'frame': overlay, 'detections': detections}

        # BGR → RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            results = self.model(img)

        # YOLOv5 output
        if not hasattr(results, 'xyxy'):
            return {'frame': overlay, 'detections': detections}

        preds = results.xyxy[0].cpu().numpy()

        for x1, y1, x2, y2, conf, cls in preds:
            if conf < self.conf_thres:
                continue

            label = f"{self.model.names[int(cls)]} {conf:.2f}"
            draw_bbox(overlay, [x1, y1, x2, y2], label=label)

            detections.append({
                'xyxy': [float(x1), float(y1), float(x2), float(y2)],
                'conf': float(conf),
                'class': int(cls),
                'label': self.model.names[int(cls)]
            })

        return {'frame': overlay, 'detections': detections}


# ----------------------------
# Local test
# ----------------------------
if __name__ == '__main__':
    od = ObstacleDetector(weights_path=None)  # safe default
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = od.detect(frame)
        cv2.imshow('obstacles', res['frame'])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
