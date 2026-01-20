"""
Simple navigation controller using lane center and obstacle presence.

Change points:
- Safety distance (pixels) can be configured in CONFIG in main.py
- Steering and throttle logic can be tuned for your camera FOV and vehicle dynamics
"""

import numpy as np
import cv2  # only for moments; local import to avoid circular dependencies

class SimpleController:
    def __init__(self, safe_distance_px=80):
        """
        :param safe_distance_px: pixel distance from bottom of frame to trigger braking
        """
        self.safe_distance_px = safe_distance_px

    def compute(self, lane_mask, detections):
        """
        Compute steering, throttle, and brake signals.

        :param lane_mask: binary mask of detected lane (H x W)
        :param detections: list of obstacles [{'xyxy': [x1,y1,x2,y2], 'conf': float, 'class': int, 'label': str}]
        :return: dict {'steer': float, 'throttle': float, 'brake': bool}
                 steer: -1 (left) .. +1 (right)
                 throttle: 0..1
                 brake: True/False
        """
        h, w = lane_mask.shape[:2]

        # Compute lane centroid
        M = None
        try:
            M = cv2.moments(lane_mask)
        except Exception:
            M = None

        if M and M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            offset = (cx - w/2) / (w/2)  # normalize -1..1
            steer = float(np.clip(offset, -1, 1))
        else:
            steer = 0.0

        # Obstacle check: brake if any obstacle is close to bottom
        brake = False
        for d in detections:
            x1, y1, x2, y2 = d['xyxy']
            if y2 >= h - self.safe_distance_px:
                brake = True
                break

        throttle = 0.5 if not brake else 0.0

        return {'steer': steer, 'throttle': throttle, 'brake': brake}


# Quick local test (optional)
if __name__ == '__main__':
    ctrl = SimpleController()
    dummy_mask = np.zeros((480,640), dtype=np.uint8)
    cv2.rectangle(dummy_mask, (270,0), (370,480), 255, -1)  # simulate lane in center
    dummy_detections = [{'xyxy':[300,400,350,470],'conf':0.9,'class':1,'label':'car'}]
    cmd = ctrl.compute(dummy_mask, dummy_detections)
    print(cmd)
