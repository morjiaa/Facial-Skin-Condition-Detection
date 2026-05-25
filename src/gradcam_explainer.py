from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from . import config
from .preprocessing import prepare_resnet_tensor
from .utils import create_output_dirs, timestamp


class GradCAMExplainer:
    def __init__(self, classifier):
        self.classifier = classifier
        self.model = classifier.model
        self.device = classifier.device
        self.target_layer = self.model.layer4[-1]
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_module, _inputs, output):
            self.activations = output.detach()

        def backward_hook(_module, _grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image, target_class: Optional[str] = None):
        create_output_dirs()
        self.model.zero_grad(set_to_none=True)

        input_tensor = prepare_resnet_tensor(image).to(self.device)
        logits = self.model(input_tensor)

        if target_class and target_class in self.classifier.class_names:
            target_idx = self.classifier.class_names.index(target_class)
        else:
            target_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, target_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        raw_cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = raw_cam
        cam = torch.relu(cam)
        if float(cam.max()) <= 0:
            cam = torch.abs(raw_cam)
        cam -= cam.min()
        if float(cam.max()) > 0:
            cam /= cam.max()

        heatmap = cam.detach().cpu().numpy()
        rgb = np.array(image.convert("RGB"))
        heatmap = cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]))

        low, high = np.percentile(heatmap, [45, 99])
        if high > low:
            heatmap = np.clip((heatmap - low) / (high - low), 0, 1)
        heatmap = np.clip(heatmap * 1.35, 0, 1)

        heatmap_uint8 = np.uint8(255 * heatmap)
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        alpha = (0.22 + 0.58 * heatmap)[..., None]
        overlay = np.uint8(np.clip(rgb * (1 - alpha) + colored * alpha, 0, 255))

        output_path = config.OUTPUT_DIR / "gradcam" / f"gradcam_{timestamp()}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return {
            "heatmap": heatmap,
            "overlay": overlay,
            "path": str(output_path),
            "target_class": self.classifier.class_names[target_idx],
        }
