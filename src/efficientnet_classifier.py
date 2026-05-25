from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision import models

from .config import RESNET_IMAGE_SIZE
from .labels import get_broad_category
from .preprocessing import prepare_resnet_tensor


class EfficientNetAcneClassifier:
    def __init__(self, model_path, device: str = "cpu"):
        self.model_path = Path(model_path)
        self.device = torch.device(device)

        if not self.model_path.exists():
            raise FileNotFoundError(f"EfficientNet acne classifier weight not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.class_names = list(checkpoint.get("class_names", []))
        if not self.class_names:
            raise ValueError("EfficientNet checkpoint does not include class_names")

        self.model = self._build_model(len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _build_model(num_classes: int):
        try:
            model = models.efficientnet_b0(weights=None)
        except TypeError:
            model = models.efficientnet_b0(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(256, num_classes),
        )
        return model

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
