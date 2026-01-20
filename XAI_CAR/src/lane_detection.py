import cv2
import numpy as np
import torch
from utils import draw_text, preprocess_for_cnn

class LaneDetector:
    def __init__(self, model_path=None, device='cpu'):
        """
        Lane Detector with CNN model and classical OpenCV fallback.
        :param model_path: Path to PyTorch CNN model (optional)
        :param device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model = None
        if model_path:
            self.model = torch.load(model_path, map_location=device)
            self.model.eval()

    def detect(self, frame):
        """
        Detect lanes in the frame.
        Returns dict: {'frame': overlay, 'lane_mask': mask_bin, 'steering_angle': angle}
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()
        angle = None
        mask_bin = None

        # ------------------------
        # CNN-based lane detection
        # ------------------------
        if self.model:
            img_tensor = torch.tensor(preprocess_for_cnn(frame), device=self.device)
            with torch.no_grad():
                pred = self.model(img_tensor)  # Modify if your CNN output differs
            mask = pred.squeeze().cpu().numpy()
            mask_bin = (mask > 0.5).astype(np.uint8) * 255
            mask_bin_resized = cv2.resize(mask_bin, (w, h))
            overlay = cv2.addWeighted(frame, 0.8, cv2.cvtColor(mask_bin_resized, cv2.COLOR_GRAY2BGR), 0.2, 0)
            # Optional: estimate angle from CNN mask (if output is segmentation)
            ys, xs = np.where(mask_bin_resized > 0)
            if len(xs) > 0:
                cx = int(xs.mean())
                angle = (cx - w//2) / (w//2) * 25  # crude steering heuristic
            if angle is not None:
                draw_text(overlay, f"Steering: {angle:.2f} deg", (10,30))
            return {'frame': overlay, 'lane_mask': mask_bin_resized, 'steering_angle': angle}

        # ------------------------
        # Classical OpenCV fallback
        # ------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Region of interest polygon
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, h),
            (w, h),
            (int(0.6*w), int(0.6*h)),
            (int(0.4*w), int(0.6*h))
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        cropped = cv2.bitwise_and(edges, mask)

        # Hough lines detection
        lines = cv2.HoughLinesP(cropped, 1, np.pi/180, 30, minLineLength=30, maxLineGap=200)
        if lines is not None:
            for x1,y1,x2,y2 in lines[:,0,:]:
                cv2.line(overlay, (x1,y1), (x2,y2), (0,255,0), 3)

            # crude steering angle heuristic: average slope
            slopes = []
            for x1,y1,x2,y2 in lines[:,0,:]:
                if x2 != x1:
                    slopes.append((y2-y1)/(x2-x1))
            if slopes:
                mean_slope = np.mean(slopes)
                angle = np.degrees(np.arctan(mean_slope))
                draw_text(overlay, f"Steering (est): {angle:.1f} deg", (10,30))

        # Lane mask (for controller/XAI)
        mask_bin = cv2.dilate(cropped, np.ones((5,5), np.uint8), iterations=2)

        return {'frame': overlay, 'lane_mask': mask_bin, 'steering_angle': angle}


# ------------------------
# Quick local test
# ------------------------
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    ld = LaneDetector(model_path=None)  # set model_path='models/lane_cnn.pth' if you have CNN
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = ld.detect(frame)
        cv2.imshow('lane', res['frame'])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
