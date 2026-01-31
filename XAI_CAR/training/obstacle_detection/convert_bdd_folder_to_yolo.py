import json
import os

# =====================================================
# PATH SETUP (ROBUST & WINDOWS-SAFE)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))

LABEL_ROOT = os.path.join(PROJECT_ROOT, "data", "bdd100k", "labels")
YOLO_LABEL_ROOT = os.path.join(PROJECT_ROOT, "data", "bdd100k", "labels_yolo")

# BDD100K image resolution (fixed)
IMG_WIDTH = 1280
IMG_HEIGHT = 720

# =====================================================
# BDD100K DETECTION CLASSES (OFFICIAL)
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
# CREATE OUTPUT DIRECTORIES
# =====================================================
for split in ["train", "val"]:
    os.makedirs(os.path.join(YOLO_LABEL_ROOT, split), exist_ok=True)

# =====================================================
# CONVERSION PROCESS
# =====================================================
total_written = 0
total_skipped = 0

for split in ["train", "val"]:
    split_label_dir = os.path.join(LABEL_ROOT, split)

    if not os.path.exists(split_label_dir):
        print(f"❌ Missing folder: {split_label_dir}")
        continue

    json_files = [f for f in os.listdir(split_label_dir) if f.endswith(".json")]
    print(f"Processing {split}: {len(json_files)} label files")

    written = 0
    skipped = 0

    for jf in json_files:
        json_path = os.path.join(split_label_dir, jf)

        with open(json_path, "r") as f:
            data = json.load(f)

        yolo_lines = []

        for obj in data.get("labels", []):
            category = obj.get("category")
            if category not in CLASSES:
                continue

            if "box2d" not in obj:
                continue

            box = obj["box2d"]
            x1, y1 = box["x1"], box["y1"]
            x2, y2 = box["x2"], box["y2"]

            # Convert to YOLO format (normalized)
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
    print(f"  → Skipped (no obstacles): {skipped}")

    total_written += written
    total_skipped += skipped

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n✅ Folder-based BDD100K → YOLO conversion COMPLETE")
print(f"Total label files written: {total_written}")
print(f"Total skipped (empty frames): {total_skipped}")
