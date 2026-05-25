from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision import models

from .config import RESNET_IMAGE_SIZE
from .labels import RESNET_CLASSES, get_broad_category
from .preprocessing import prepare_resnet_tensor


class ResNetClassifier:
    def __init__(self, model_path, class_names: List[str] = None, device: str = "cpu"):
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.class_names = class_names or list(RESNET_CLASSES)

        if not self.model_path.exists():
            raise FileNotFoundError(f"ResNet18 model weight not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and checkpoint.get("class_names"):
            self.class_names = list(checkpoint["class_names"])

        self.model = self._build_model(len(self.class_names))
        state_dict = self._extract_state_dict(checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _build_model(num_classes: int):
        try:
            model = models.resnet18(weights=None)
        except TypeError:
            model = models.resnet18(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, num_classes))
        return model

    @staticmethod
    def _extract_state_dict(checkpoint):
        if isinstance(checkpoint, dict):
            for key in ["model_state_dict", "state_dict"]:
                if key in checkpoint:
                    return checkpoint[key]
        return checkpoint

    @torch.no_grad()
    def predict(self, image, top_k: int = 3) -> Dict:
        input_tensor = prepare_resnet_tensor(image, RESNET_IMAGE_SIZE).to(self.device)
        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        k = min(top_k, len(self.class_names))
        top_probs, top_indices = torch.topk(probabilities, k=k)

        top_k_predictions = []
        for prob, idx in zip(top_probs.detach().cpu().tolist(), top_indices.detach().cpu().tolist()):
            class_name = self.class_names[int(idx)]
            top_k_predictions.append(
                {
                    "class_name": class_name,
                    "probability": float(prob),
                    "broad_category": get_broad_category(class_name),
                }
            )

        predicted_class = top_k_predictions[0]["class_name"]
        return {
            "predicted_class": predicted_class,
            "probability": top_k_predictions[0]["probability"],
            "top_k": top_k_predictions,
            "broad_category": get_broad_category(predicted_class),
            "logits": logits.detach().cpu(),
            "input_tensor": input_tensor.detach().cpu(),
        }
