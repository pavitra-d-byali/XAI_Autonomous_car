"""
Main integration script for XAI Autonomous Car
Clean version (UNet + YOLOv8)
"""

import argparse
import os
import time
import cv2
import numpy as np

from lane_detection import LaneDetector
from obstacle_detection import ObstacleDetector
from controller import SimpleController
from utils import resize_keep_aspect, draw_text


# ============================================================
# PATH SETUP (ABSOLUTE & SAFE)
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

CONFIG = {
    # ✅ Correct checkpoint path
    "lane_model_path": os.path.join(
        BASE_DIR,
        "models",
        "checkpoints",
        "lane_unet_epoch_5.pth"
    ),

    "yolo_weights": os.path.join(BASE_DIR, "models", "yolov8n.pt"),

    "device": "cpu",   # Change to "cuda" if GPU available
    "output_dir": os.path.join(BASE_DIR, "outputs"),
    "save_video": True,
    "show": True,
}
# ============================================================


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--camera", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(CONFIG["output_dir"])

    # --------------------------------------------------------
    # Initialize modules
    # --------------------------------------------------------
    lane_detector = LaneDetector(CONFIG["lane_model_path"], CONFIG["device"])
    obstacle_detector = ObstacleDetector(CONFIG["yolo_weights"], CONFIG["device"])
    controller = SimpleController(safe_distance_px=80)

    # --------------------------------------------------------
    # Video source
    # --------------------------------------------------------
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        raise ValueError("Use --video or --camera")

    if not cap.isOpened():
        raise RuntimeError("Could not open video source")

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------
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

        frame_proc = resize_keep_aspect(frame, width=640)

        # -------- Lane Detection --------
        lane_res = lane_detector.detect(frame_proc)

        # -------- Obstacle Detection --------
        obs_res = obstacle_detector.detect(frame_proc)

        # -------- Merge Visualization --------
        vis = lane_res["frame"].copy()

        if obs_res["frame"] is not None:
            yolo_frame = obs_res["frame"]
            mask = (yolo_frame != frame_proc).any(axis=2)
            vis[mask] = yolo_frame[mask]

        # -------- Controller --------
        lane_mask = (
            lane_res["lane_mask"]
            if lane_res["lane_mask"] is not None
            else np.zeros(frame_proc.shape[:2], dtype=np.uint8)
        )

        detections = obs_res.get("detections", [])
        commands = controller.compute(lane_mask, detections)

        # -------- HUD --------
        draw_text(vis, f"Steer: {commands['steer']:.2f}", (10, 25))
        draw_text(vis, f"Throttle: {commands['throttle']:.2f}", (10, 50))
        draw_text(vis, f"Brake: {commands['brake']}", (10, 75))

        # -------- Display --------
        if CONFIG["show"]:
            cv2.imshow("XAI Autonomous Car", vis)

        # -------- Save --------
        if CONFIG["save_video"]:
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))
                print(f"[INFO] Saving output to {out_path}")
            writer.write(vis)

        frame_count += 1

        if CONFIG["show"] and cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = round(time.time() - start_time, 2)
    print(f"[DONE] Processed {frame_count} frames in {elapsed} sec")


if __name__ == "__main__":
    main()
 