import os
import json
import cv2
import numpy as np

def create_mask(image_path, lanes, h_samples, save_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for lane in lanes:
        points = []
        for x, y in zip(lane, h_samples):
            if x >= 0:
                points.append((x, y))

        if len(points) > 1:
            for i in range(len(points)-1):
                cv2.line(mask, points[i], points[i+1], 255, 5)

    cv2.imwrite(save_path, mask)

def process(json_file, root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        data = json.loads(line.strip())
        image_path = os.path.join(root_dir, data["raw_file"])
        lanes = data["lanes"]
        h_samples = data["h_samples"]

        filename = os.path.basename(image_path)
        save_path = os.path.join(output_dir, filename.replace(".jpg", ".png"))

        create_mask(image_path, lanes, h_samples, save_path)

if __name__ == "__main__":
    process(
        "data/tusimple/label_data_0313.json",
        "data/tusimple",
        "data/tusimple_masks"
    )
