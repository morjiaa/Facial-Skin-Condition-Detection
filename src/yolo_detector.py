from pathlib import Path
from typing import Dict, List

from .labels import YOLO_CLASSES, get_broad_category


class YOLODetector:
    def __init__(self, model_path, conf: float, iou: float, device: str):
        self.model_path = Path(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.model = None
        self.names = {idx: name for idx, name in enumerate(YOLO_CLASSES)}

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model weight not found: {self.model_path}")

        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))
        if getattr(self.model, "names", None):
            self.names = self.model.names

    def predict(self, image) -> List[Dict]:
        results = self.model.predict(image, conf=self.conf, iou=self.iou, device=self.device, verbose=False)
        if not results:
            return []

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)

        detections = []
        for box, confidence, class_id in zip(xyxy, confs, classes):
            class_name = self.names.get(int(class_id), YOLO_CLASSES[int(class_id)] if int(class_id) < len(YOLO_CLASSES) else str(class_id))
            detections.append(
                {
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "box_xyxy": [float(v) for v in box.tolist()],
                    "broad_category": get_broad_category(class_name),
                }
            )

        return sorted(detections, key=lambda item: item["confidence"], reverse=True)
