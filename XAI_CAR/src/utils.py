"""Helper utilities: drawing, preprocessing, and I/O.
Change points: set colors, font scales, or debug flags below.
"""
import cv2
import numpy as np

# Visualization constants
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_text(img, text, pos=(10, 30), color=(0, 255, 0), scale=0.8, thickness=2):
    cv2.putText(img, text, pos, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_bbox(img, xyxy, label=None, color=(0, 0, 255), thickness=2):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        txt_size = cv2.getTextSize(label, FONT, 0.6, 1)[0]
        cv2.rectangle(img, (x1, y1 - txt_size[1] - 6),
                      (x1 + txt_size[0] + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4),
                    FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def resize_keep_aspect(img, width=None, height=None):
    h, w = img.shape[:2]
    if width is None and height is None:
        return img
    if width is None:
        scale = height / h
        width = int(w * scale)
    else:
        scale = width / w
        height = int(h * scale)
    return cv2.resize(img, (width, height))


def preprocess_for_cnn(img, size=(224, 224)):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, size)
    arr = np.float32(img_resized) / 255.0
    # HWC -> CHW
    arr = arr.transpose(2, 0, 1)
    return arr
