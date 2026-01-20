"""
Main integration script for XAI Autonomous Car.

Usage examples:
    python src/main.py --video ../data/test_videos/sample.mp4
    python src/main.py --camera 0

Change points:
- Set model paths, device, output directory in CONFIG.
- Adjust Grad-CAM target layer according to your lane CNN architecture.
"""

import argparse
import os
import time
import cv2
import numpy as np
import torch

from lane_detection import LaneDetector
from obstacle_detection import ObstacleDetector
from xai_module import GradCAM
from controller import SimpleController
from utils import resize_keep_aspect, draw_text, preprocess_for_cnn


# ---------------- CONFIG (edit these) ----------------
CONFIG = {
    'lane_model_path': 'models/lane_cnn.pth',    # path to lane CNN
    'yolo_weights': 'models/yolov5_tiny.pt',    # YOLOv5 weights
    'device': 'cpu',                             # 'cpu' or 'cuda'
    'output_dir': 'outputs',
    'save_video': True,
    'show': True,
    'camera_index': None,
    'xai_target_layer': 'features.7'             # change to match lane CNN architecture
}
# ---------------------------------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default=None, help='Path to input video')
    parser.add_argument('--camera', type=int, default=None, help='Camera index (e.g., 0)')
    return parser.parse_args()


def main():
    args = parse_args()
    device = CONFIG['device']
    ensure_dir(CONFIG['output_dir'])

    # Initialize modules
    lane_detector = LaneDetector(model_path=CONFIG['lane_model_path'], device=device)
    obstacle_detector = ObstacleDetector(weights_path=CONFIG['yolo_weights'], device=device)
    controller = SimpleController(safe_distance_px=80)

    # Setup Grad-CAM for lane CNN (optional)
    gradcam = None
    if lane_detector.use_cnn:
        try:
            gradcam = GradCAM(lane_detector.model, CONFIG['xai_target_layer'])
            print('GradCAM ready')
        except Exception as e:
            print('GradCAM setup failed:', e)

    # Video capture
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        raise ValueError('Provide --video <path> or --camera <index>')

    # Video writer setup
    writer = None
    if CONFIG['save_video']:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.join(CONFIG['output_dir'], 'run_output.mp4')

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for speed
        frame_proc = resize_keep_aspect(frame, width=640)

        # ---------------- Lane Detection ----------------
        lane_res = lane_detector.detect(frame_proc)

        # ---------------- Obstacle Detection ----------------
        obs_res = obstacle_detector.detect(frame_proc)

        # ---------------- XAI Overlay ----------------
        if gradcam and lane_detector.use_cnn:
            inp = torch.from_numpy(preprocess_for_cnn(frame_proc, size=(224,224)).astype('float32')).unsqueeze(0).to(device)
            try:
                heatmap = gradcam.generate(inp)
                frame_with_xai = cv2.addWeighted(frame_proc.copy(), 0.5, heatmap, 0.5, 0)

            except Exception as e:
                print('GradCAM generate failed:', e)
                frame_with_xai = lane_res['frame']
        else:
            frame_with_xai = lane_res['frame']

        # ---------------- Controller ----------------
        lane_mask = lane_res['lane_mask'] if lane_res['lane_mask'] is not None else np.zeros(frame_proc.shape[:2], dtype='uint8')
        detections = obs_res.get('detections', [])
        commands = controller.compute(lane_mask, detections)

        # ---------------- Visualization ----------------
        vis = frame_with_xai.copy()
        # Overlay obstacle detection
        vis = obs_res['frame']

        # HUD
        draw_text(vis, f"Steer: {commands['steer']:.2f}", (10,25))
        draw_text(vis, f"Throttle: {commands['throttle']:.2f}", (10,50))
        draw_text(vis, f"Brake: {commands['brake']}", (10,75))

        # Show frame
        if CONFIG['show']:
            cv2.imshow('XAI_Autonomous_Car', vis)

        # Save video
        if CONFIG['save_video']:
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(out_path, fourcc, 20.0, (w,h))
            writer.write(vis)

        frame_idx += 1
        if CONFIG['show'] and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    if CONFIG['show']:
        cv2.destroyAllWindows()
    print('Done. Processed', frame_idx, 'frames in', round(time.time()-t0,2), 'seconds')


if __name__ == '__main__':
    main()
