"""
Simple Grad-CAM integration for PyTorch models.
Change points: specify target layer name in use.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        """
        :param model: PyTorch CNN model
        :param target_layer: str, name of the layer to visualize
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # register hooks
        layer = dict(self.model.named_modules())[target_layer]
        layer.register_forward_hook(self._forward_hook)
        layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        :param input_tensor: torch tensor of shape (1,C,H,W)
        :param class_idx: int, target class index for Grad-CAM
        :return: heatmap (H,W) normalized 0-255
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        # Gradients w.r.t. target class
        score = output[0, class_idx]
        score.backward(retain_graph=True)

        # Pool gradients across channels
        pooled_grads = torch.mean(self.gradients, dim=(2,3))  # (B,C)
        activations = self.activations[0]  # (C,H,W)

        # Weight activations by pooled gradients
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_grads[0, i]

        # Sum channels
        heatmap = torch.sum(activations, dim=0)
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap) + 1e-8  # normalize 0-1
        heatmap = heatmap.cpu().numpy()
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap

# ------------------------
# Quick usage example
# ------------------------
if __name__ == '__main__':
    # Example:
    # from lane_detection import LaneDetector
    # ld = LaneDetector(model_path='models/lane_cnn.pth')
    # gc = GradCAM(ld.model, target_layer='features.7')
    # frame_tensor = torch.tensor(preprocess_for_cnn(frame))
    # heatmap = gc.generate(frame_tensor)
    pass
