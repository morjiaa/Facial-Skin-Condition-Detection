from PIL import Image

from src.decision_engine import HybridDecisionEngine


class MockYOLO:
    def __init__(self, detections):
        self.detections = detections

    def predict(self, _image):
        return self.detections


class MockResNet:
    def __init__(self):
        self.used = False

    def predict(self, _image, top_k=3):
        self.used = True
        return {
            "predicted_class": "Papules",
            "probability": 0.7,
            "top_k": [
                {"class_name": "Papules", "probability": 0.7, "broad_category": "Acne-spectrum lesions"}
            ],
            "broad_category": "Acne-spectrum lesions",
        }


class MockGradCAM:
    def generate(self, _image, target_class=None):
        return {"path": "mock.png", "overlay": None}


def detection(confidence):
    return {
        "class_name": "Papular",
        "confidence": confidence,
        "box_xyxy": [0, 0, 40, 40],
        "broad_category": "Acne-spectrum lesions",
    }


def run_case(detections):
    image = Image.new("RGB", (100, 100), color="white")
    resnet = MockResNet()
    engine = HybridDecisionEngine(MockYOLO(detections), resnet, MockGradCAM())
    return engine.run(image), resnet


def test_high_confidence_yolo_does_not_use_resnet():
    result, resnet = run_case([detection(0.80)])
    assert result["result_type"] == "YOLO_HIGH_CONFIDENCE_DETECTION"
    assert resnet.used is False


def test_low_confidence_yolo_uses_resnet():
    result, resnet = run_case([detection(0.50)])
    assert result["result_type"] == "LOW_CONFIDENCE_YOLO_WITH_RESNET_FALLBACK"
    assert resnet.used is True


def test_no_yolo_detection_uses_resnet():
    result, resnet = run_case([])
    assert result["result_type"] == "NO_YOLO_DETECTION_RESNET_FALLBACK"
    assert resnet.used is True


def test_mixed_confidence_uses_high_confidence_path():
    result, resnet = run_case([detection(0.80), detection(0.40)])
    assert result["result_type"] == "YOLO_HIGH_CONFIDENCE_DETECTION"
    assert resnet.used is False
