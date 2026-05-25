from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


BASE_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = BASE_DIR / "weights"
OUTPUT_DIR = BASE_DIR / "outputs"

YOLO_MODEL_PATH = WEIGHTS_DIR / "best.pt"
RESNET_MODEL_PATH = WEIGHTS_DIR / "best_model.pt"
CLASSMATE_YOLO_MODEL_PATH = BASE_DIR / "classmate" / "acne_model.pt"
CLASSMATE_CLASSIFIER_MODEL_PATH = BASE_DIR / "classmate" / "acne_classifier.pth"
USE_CLASSMATE_ACNESCAN_PIPELINE = False

RELIABLE_YOLO_CONFIDENCE = 0.65
RAW_YOLO_CONFIDENCE = 0.05
YOLO_IOU_THRESHOLD = 0.45
RESNET_IMAGE_SIZE = 224
TOP_K = 3
LOW_CONFIDENCE_CROP_PADDING = 0.35
ACNESCAN_CROP_PADDING = 0.35
ACNESCAN_BOX_CONFIDENCE = 0.05

DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

DISCLAIMER = "For educational demonstration only — not a medical diagnosis tool."
