from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DetectionResult:
    class_name: str
    confidence: float
    box_xyxy: List[float]
    broad_category: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "confidence": self.confidence,
            "box_xyxy": self.box_xyxy,
            "broad_category": self.broad_category,
        }


@dataclass
class TopKPrediction:
    class_name: str
    probability: float
    broad_category: str


@dataclass
class ClassificationResult:
    predicted_class: str
    probability: float
    top_k: List[TopKPrediction]
    broad_category: str
    logits: Optional[Any] = None
    input_tensor: Optional[Any] = None


@dataclass
class HybridInferenceResult:
    result_type: str
    model_source: str
    visualization_type: str
    yolo: Dict[str, Any]
    resnet: Dict[str, Any]
    final_summary: Dict[str, Any]
    disclaimer: str
    status_messages: List[str] = field(default_factory=list)
