import cv2
import numpy as np
import torch

from utils import draw_text, preprocess_for_cnn
from training.lane_unet import LaneUNet


class LaneDetector:
    """
    Lane Detector using:
    - UNet segmentation (primary)
    - OpenCV fallback (safe mode)
    """

    def __init__(self, model_path=None, device="cpu"):
        self.device = device
        self.model = None
        self.use_cnn = False

        if model_path is not None:
            try:
                self.model = LaneUNet().to(self.device)
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()

                self.use_cnn = True
                print("[INFO] Lane UNet loaded successfully")

            except Exception as e:
                print("[WARN] UNet load failed, using OpenCV fallback:", e)
                self.use_cnn = False

    # ==========================================================
    def detect(self, frame):

        h, w = frame.shape[:2]
        overlay = frame.copy()
        steering_angle = None
        lane_mask = None

        # ======================================================
        # 1️⃣ UNet Segmentation
        # ======================================================
        if self.use_cnn and self.model is not None:
            try:
                inp = preprocess_for_cnn(frame, size=(256, 512))
                inp = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = self.model(inp)
                    probs = torch.sigmoid(logits)

                mask = probs.squeeze().cpu().numpy()

                # 🔥 Lower threshold (important)
                lane_mask = (mask > 0.3).astype(np.uint8) * 255
                lane_mask = cv2.resize(lane_mask, (w, h))

                # Only consider bottom 40% of image
                roi_start = int(h * 0.6)
                roi_mask = lane_mask[roi_start:h, :]

                ys, xs = np.where(roi_mask > 0)

                if len(xs) > 200:

                    # Split into left/right by image center
                    left_pixels = xs[xs < w // 2]
                    right_pixels = xs[xs >= w // 2]

                    left_x = None
                    right_x = None

                    if len(left_pixels) > 100:
                        left_x = int(np.mean(left_pixels))

                    if len(right_pixels) > 100:
                        right_x = int(np.mean(right_pixels))

                    # Draw lanes if detected
                    if left_x is not None:
                        cv2.line(
                            overlay,
                            (left_x, h),
                            (left_x, roi_start),
                            (0, 255, 0),
                            3,
                        )

                    if right_x is not None:
                        cv2.line(
                            overlay,
                            (right_x, h),
                            (right_x, roi_start),
                            (0, 255, 0),
                            3,
                        )

                    if left_x is not None and right_x is not None:
                        center_x = int((left_x + right_x) / 2)

                        cv2.line(
                            overlay,
                            (center_x, h),
                            (center_x, roi_start),
                            (0, 0, 255),
                            3,
                        )

                        steering_angle = (center_x - w // 2) / (w // 2) * 25

                # Draw mask overlay
                color_mask = np.zeros_like(frame)
                color_mask[lane_mask > 0] = [0, 255, 0]
                overlay = cv2.addWeighted(overlay, 0.8, color_mask, 0.3, 0)

            except Exception as e:
                print("[WARN] UNet inference failed:", e)
                self.use_cnn = False

        # ======================================================
        # 2️⃣ OpenCV fallback
        # ======================================================
        if not self.use_cnn:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)

            roi = np.zeros_like(edges)
            polygon = np.array(
                [[
                    (0, h),
                    (w, h),
                    (int(0.6 * w), int(0.6 * h)),
                    (int(0.4 * w), int(0.6 * h)),
                ]],
                np.int32,
            )

            cv2.fillPoly(roi, polygon, 255)
            cropped = cv2.bitwise_and(edges, roi)

            lines = cv2.HoughLinesP(
                cropped,
                rho=1,
                theta=np.pi / 180,
                threshold=50,
                minLineLength=50,
                maxLineGap=150,
            )

            left_x = []
            right_x = []

            if lines is not None:
                for x1, y1, x2, y2 in lines[:, 0]:
                    if x2 == x1:
                        continue

                    slope = (y2 - y1) / (x2 - x1)

                    if abs(slope) < 0.5:
                        continue

                    if slope < 0:
                        left_x.append(x1)
                    else:
                        right_x.append(x1)

                    cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)

            if left_x and right_x:
                lx = int(np.mean(left_x))
                rx = int(np.mean(right_x))
                center_x = int((lx + rx) / 2)

                cv2.line(
                    overlay,
                    (center_x, h),
                    (center_x, int(h * 0.6)),
                    (0, 0, 255),
                    3,
                )

                steering_angle = (center_x - w // 2) / (w // 2) * 25

            lane_mask = cropped

        # ======================================================
        # Steering Text
        # ======================================================
        if steering_angle is not None:
            draw_text(overlay, f"Steering: {steering_angle:.2f}", (10, 30))
        else:
            draw_text(overlay, "Steering: N/A", (10, 30))

        return {
            "frame": overlay,
            "lane_mask": lane_mask,
            "steering_angle": steering_angle,
        }
