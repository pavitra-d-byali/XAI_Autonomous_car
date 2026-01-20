"""Obstacle detection module using YOLOv5 (torch.hub) or a saved yolov5_tiny.pt custom weight.
Change points: set `weights_path` in initialization. If you prefer to use local yolov5 clone, replace torch.hub usage.
"""
import torch
import cv2
import numpy as np
from utils import draw_bbox


class ObstacleDetector:
    def __init__(self, weights_path=None, device='cpu', conf_thres=0.3):
        self.device = device
        self.weights_path = weights_path
        self.conf_thres = conf_thres
        # Try to load with torch.hub (ultralytics)
        try:
            if weights_path and weights_path.endswith('.pt'):
                # custom weights
                self.model = torch.hub.load(
                    'ultralytics/yolov5',
                    'custom',
                    path=weights_path,
                    force_reload=False
                )
            else:
                # fallback to pretrained small
                self.model = torch.hub.load(
                    'ultralytics/yolov5',
                    'yolov5s',
                    pretrained=True
                )
            self.model.to(self.device)
            self.model.conf = conf_thres
            print('YOLOv5 model loaded')
        except Exception as e:
            raise RuntimeError(
                'Failed to load YOLOv5 via torch.hub. '
                'Make sure internet access is allowed or use local yolov5 clone. '
                f'Error: {e}'
            )

    def detect(self, frame):
        # Convert BGR->RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(img)

        # results.xyxy[0] -> tensor N x 6 (x1,y1,x2,y2,conf,class)
        preds = results.xyxy[0].cpu().numpy() if hasattr(results, 'xyxy') else []
        detections = []
        overlay = frame.copy()

        for *xyxy, conf, cls in preds:
            label = f"{self.model.names[int(cls)]} {conf:.2f}"
            draw_bbox(overlay, xyxy, label=label)
            detections.append({
                'xyxy': [float(x) for x in xyxy],
                'conf': float(conf),
                'class': int(cls),
                'label': self.model.names[int(cls)]
            })

        return {'frame': overlay, 'detections': detections}


if __name__ == '__main__':
    od = ObstacleDetector(weights_path='models/yolov5_tiny.pt')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = od.detect(frame)
        cv2.imshow('obs', res['frame'])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
