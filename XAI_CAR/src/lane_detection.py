import cv2
import numpy as np
import torch

from utils import draw_text, preprocess_for_cnn
from lane_model import LaneCNN


class LaneDetector:
    """
    Lane Detector using:
    - CNN-based lane segmentation (preferred)
    - Classical OpenCV fallback (safe mode)
    """

    def __init__(self, model_path=None, device="cpu"):
        self.device = device
        self.model = None
        self.use_cnn = False

        if model_path:
            try:
                # Initialize architecture
                self.model = LaneCNN().to(self.device)

                # Load WEIGHTS ONLY (PyTorch 2.6+ safe)
                state_dict = torch.load(
                    model_path,
                    map_location=self.device,
                    weights_only=True
                )
                self.model.load_state_dict(state_dict)
                self.model.eval()

                self.use_cnn = True
                print("[INFO] Lane CNN loaded successfully")

            except Exception as e:
                print("[WARN] CNN load failed, using OpenCV fallback:", e)
                self.model = None
                self.use_cnn = False

    def detect(self, frame):
        """
        Args:
            frame (BGR image)

        Returns:
            dict:
                frame           -> visualization
                lane_mask       -> binary mask
                steering_angle  -> float | None
        """

        h, w = frame.shape[:2]
        overlay = frame.copy()
        lane_mask = None
        steering_angle = None

        # =====================================================
        # CNN-based lane detection
        # =====================================================
        if self.use_cnn and self.model is not None:
            try:
                inp = preprocess_for_cnn(frame, size=(224, 224))
                inp = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = self.model(inp)
                    probs = torch.sigmoid(logits)

                mask = probs.squeeze().cpu().numpy()
                lane_mask = (mask > 0.5).astype(np.uint8) * 255
                lane_mask = cv2.resize(lane_mask, (w, h))

                # Overlay visualization
                overlay = cv2.addWeighted(
                    frame, 0.8,
                    cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR),
                    0.2, 0
                )

                # Steering estimation
                ys, xs = np.where(lane_mask > 0)
                if len(xs) > 0:
                    center_x = int(xs.mean())
                    steering_angle = (center_x - w // 2) / (w // 2) * 25
                    draw_text(overlay, f"Steer: {steering_angle:.2f}", (10, 30))

                return {
                    "frame": overlay,
                    "lane_mask": lane_mask,
                    "steering_angle": steering_angle
                }

            except Exception as e:
                print("[WARN] CNN inference failed, switching to OpenCV:", e)
                self.use_cnn = False

        # =====================================================
        # Classical OpenCV fallback
        # =====================================================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        roi = np.zeros_like(edges)
        polygon = np.array([[ 
            (0, h),
            (w, h),
            (int(0.6 * w), int(0.6 * h)),
            (int(0.4 * w), int(0.6 * h))
        ]], np.int32)

        cv2.fillPoly(roi, polygon, 255)
        cropped = cv2.bitwise_and(edges, roi)

        lines = cv2.HoughLinesP(
            cropped, 1, np.pi / 180,
            threshold=30,
            minLineLength=30,
            maxLineGap=200
        )

        slopes = []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
                if x2 != x1:
                    slopes.append((y2 - y1) / (x2 - x1))

        if slopes:
            mean_slope = np.mean(slopes)
            steering_angle = np.degrees(np.arctan(mean_slope))
            draw_text(overlay, f"Steer(est): {steering_angle:.1f}", (10, 30))

        lane_mask = cv2.dilate(cropped, np.ones((5, 5), np.uint8), iterations=2)

        return {
            "frame": overlay,
            "lane_mask": lane_mask,
            "steering_angle": steering_angle
        }
