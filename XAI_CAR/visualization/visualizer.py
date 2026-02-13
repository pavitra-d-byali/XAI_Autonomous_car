import cv2
import numpy as np

def overlay_cam(image, cam):
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return overlay