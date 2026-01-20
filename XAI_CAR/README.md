# XAI_Autonomous_Car


Structure and files included. Replace model weight paths in `models/` and update `CONFIG` in `main.py` as needed.


Quick start:
1. Install dependencies: `pip install -r requirements.txt`
2. Put your `lane_cnn.pth` and `yolov5_tiny.pt` into `models/`.
3. Run: `python src/main.py --video data/test_videos/sample.mp4` (or use camera `--camera 0`).


Notes on change points are in each file header.