"""
Main integration script for XAI Autonomous Car
Advanced Version (UNet Lane + YOLOv8 + Controller)
"""

import argparse
import os
import time
import cv2
import numpy as np


from src.lane_detection import LaneDetector
from src.obstacle_detection import ObstacleDetector
from src.controller import SimpleController
from src.utils import resize_keep_aspect, draw_text



# ============================================================
# BASE PATH
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

CONFIG = {
    # 🔥 Always use BEST model
    "lane_model_path": os.path.join(
        BASE_DIR,
        "models",
        "checkpoints",
        "lane_unet_best.pth"
    ),

    "yolo_weights": os.path.join(BASE_DIR, "models", "yolov8n.pt"),

    "device": "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",

    "output_dir": os.path.join(BASE_DIR, "outputs"),
    "save_video": True,
    "show": True,
}
# ============================================================


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--camera", type=int, help="Camera index")
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================
def main():

    args = parse_args()
    ensure_dir(CONFIG["output_dir"])

    print("[INFO] Using device:", CONFIG["device"])

    # ---------------- Initialize Modules ----------------
    lane_detector = LaneDetector(CONFIG["lane_model_path"], CONFIG["device"])
    obstacle_detector = ObstacleDetector(CONFIG["yolo_weights"], CONFIG["device"])
    controller = SimpleController(safe_distance_px=80)

    # ---------------- Video Source ----------------
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        raise ValueError("Provide --video or --camera")

    if not cap.isOpened():
        raise RuntimeError("Failed to open video source")

    # ---------------- Video Writer ----------------
    writer = None
    out_path = os.path.join(CONFIG["output_dir"], "run_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    frame_count = 0
    start_time = time.time()

    # ========================================================
    # MAIN LOOP
    # ========================================================
    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = resize_keep_aspect(frame, width=640)

        # ===============================
        # LANE DETECTION
        # ===============================
        lane_res = lane_detector.detect(frame)

        # ===============================
        # OBSTACLE DETECTION
        # ===============================
        obs_res = obstacle_detector.detect(frame)

        # ===============================
        # VISUAL MERGE
        # ===============================
        vis = lane_res["frame"]

        if obs_res["frame"] is not None:
            yolo_overlay = obs_res["frame"]
            mask = (yolo_overlay != frame).any(axis=2)
            vis[mask] = yolo_overlay[mask]

        # ===============================
        # CONTROLLER
        # ===============================
        lane_mask = lane_res.get("lane_mask")
        if lane_mask is None:
            lane_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        detections = obs_res.get("detections", [])
        commands = controller.compute(lane_mask, detections)

        # ===============================
        # HUD
        # ===============================
        draw_text(vis, f"Steer: {commands['steer']:.2f}", (10, 25))
        draw_text(vis, f"Throttle: {commands['throttle']:.2f}", (10, 50))
        draw_text(vis, f"Brake: {commands['brake']}", (10, 75))

        # ===============================
        # FPS COUNTER
        # ===============================
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        draw_text(vis, f"FPS: {fps:.2f}", (10, 100))

        # ===============================
        # DISPLAY
        # ===============================
        if CONFIG["show"]:
            cv2.imshow("XAI Autonomous Car", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # ===============================
        # SAVE VIDEO
        # ===============================
        if CONFIG["save_video"]:
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))
                print("[INFO] Saving output to:", out_path)
            writer.write(vis)

    # ========================================================
    # CLEANUP
    # ========================================================
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    total_time = round(time.time() - start_time, 2)
    print(f"[DONE] Processed {frame_count} frames in {total_time} sec")
    print(f"[INFO] Avg FPS: {frame_count / total_time:.2f}")


if __name__ == "__main__":
    main()
