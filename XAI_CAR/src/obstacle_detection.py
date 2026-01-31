import json
import os

# =====================================================
# PATH SETUP
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))

LABEL_ROOT = os.path.join(PROJECT_ROOT, "data", "bdd100k", "labels")
YOLO_LABEL_ROOT = os.path.join(PROJECT_ROOT, "data", "bdd100k", "labels_yolo")

IMG_WIDTH = 1280
IMG_HEIGHT = 720

# =====================================================
# CLASSES (BDD100K OFFICIAL)
# =====================================================
CLASSES = {
    "person": 0,
    "rider": 1,
    "car": 2,
    "bus": 3,
    "truck": 4,
    "bike": 5,
    "motor": 6,
    "traffic light": 7,
    "traffic sign": 8
}

# =====================================================
# CREATE OUTPUT DIRS
# =====================================================
for split in ["train", "val"]:
    os.makedirs(os.path.join(YOLO_LABEL_ROOT, split), exist_ok=True)

# =====================================================
# CONVERSION
# =====================================================
total_written = 0
total_skipped = 0

for split in ["train", "val"]:
    split_dir = os.path.join(LABEL_ROOT, split)
    json_files = [f for f in os.listdir(split_dir) if f.endswith(".json")]

    print(f"Processing {split}: {len(json_files)} label files")

    written = 0
    skipped = 0

    for jf in json_files:
        json_path = os.path.join(split_dir, jf)

        with open(json_path, "r") as f:
            data = json.load(f)

        yolo_lines = []

        # 🔥 KEY FIX: iterate FRAMES → OBJECTS
        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):

                category = obj.get("category")
                if category not in CLASSES:
                    continue

                box = obj.get("box2d")
                if box is None:
                    continue

                x1, y1 = box["x1"], box["y1"]
                x2, y2 = box["x2"], box["y2"]

                if x2 <= x1 or y2 <= y1:
                    continue

                x_center = ((x1 + x2) / 2) / IMG_WIDTH
                y_center = ((y1 + y2) / 2) / IMG_HEIGHT
                width = (x2 - x1) / IMG_WIDTH
                height = (y2 - y1) / IMG_HEIGHT

                class_id = CLASSES[category]

                yolo_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

        if yolo_lines:
            txt_name = jf.replace(".json", ".txt")
            out_path = os.path.join(YOLO_LABEL_ROOT, split, txt_name)

            with open(out_path, "w") as f:
                f.write("\n".join(yolo_lines))

            written += 1
        else:
            skipped += 1

    print(f"  → Written files: {written}")
    print(f"  → Skipped (empty): {skipped}")

    total_written += written
    total_skipped += skipped

print("\n✅ BDD100K VIDEO-style → YOLO conversion COMPLETE")
print(f"Total written: {total_written}")
print(f"Total skipped: {total_skipped}")
