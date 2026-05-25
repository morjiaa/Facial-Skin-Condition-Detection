# AI-Based Facial Skin Condition Detection Using Convolutional Neural Networks

## 1. Project Overview

This is a deep learning course demo with a custom frontend in `Facial-Skin-Condition-Detection/` and a FastAPI inference backend in `server.py`. It implements a confidence-aware hybrid inference pipeline:

- YOLOv8m is the primary object detection model and is the only model that outputs bounding boxes.
- ResNet18 is used only as a fallback auxiliary image-level classifier when YOLOv8m has no reliable detection or only low-confidence candidate detections.
- Grad-CAM explains ResNet18 classification decisions by highlighting image regions that influenced the classifier.

The application does not force every image through both models. If YOLOv8m provides reliable detections, ResNet18 fallback is skipped.

## 2. Architecture

The project uses three datasets with different label spaces:

- Dataset A: YOLOv8m object detection dataset with 15 bounding-box classes.
- Dataset B: ResNet18 broad skin condition classification dataset with 6 classes.
- Dataset C: ResNet18 acne subtype classification dataset with 5 classes.

The training labels are not forced into one unified class list. Result presentation uses semantic label mapping and broad-category grouping only where classes overlap.

Default ResNet18 class order:

```python
[
    "Acne",
    "Carcinoma",
    "Eczema",
    "Keratosis",
    "Milia",
    "Rosacea",
    "Blackheads",
    "Cyst",
    "Papules",
    "Pustules",
    "Whiteheads",
]
```

Warning: Please update `RESNET_CLASSES` in `src/labels.py` if your trained ResNet18 model used a different class order.

## 3. How to Run

```bash
cd /data/shcui/CNN/facial-skin-condition-detection
pip install -r requirements.txt
uvicorn server:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

The frontend calls the backend endpoint at `POST /api/analyze`. Generated upload, YOLO, and Grad-CAM images are served from `/outputs/...`.

If the expected model weights are not present, the backend automatically uses demo mode with mock predictions so the frontend can be previewed. Demo mode is for UI testing only and is not real model inference.

## 4. Model Weights

Expected default paths:

```text
weights/yolov8m_skin_best.pt
weights/resnet18_skin_best.pth
```

You can edit default paths in `src/config.py`.

The current parent `CNN` folder contains trained ResNet checkpoints under `../outputs/`, but those checkpoints are 3-class, 5-class, or 6-class models. They are not the 11-class flat ResNet model described by this app unless you train/export one separately.

## 5. Important Configuration

Main thresholds are defined in `src/config.py`:

```python
RELIABLE_YOLO_CONFIDENCE = 0.65
RAW_YOLO_CONFIDENCE = 0.05
YOLO_IOU_THRESHOLD = 0.45
RESNET_IMAGE_SIZE = 224
TOP_K = 3
```

YOLO inference intentionally runs with `RAW_YOLO_CONFIDENCE = 0.05`, not `0.65`, so low-confidence candidate boxes remain available for ResNet18 fallback decisions while filtering out near-zero confidence noise.

Preprocessing should match the preprocessing used during model training. The current ResNet18 inference transform uses ImageNet normalization and resizes images to `224 x 224`.

## 6. Output Types

The decision engine returns one of three paths:

- `YOLO_HIGH_CONFIDENCE_DETECTION`: YOLOv8m produced at least one detection with confidence >= `0.65`. Bounding boxes are shown and ResNet18 fallback is not used.
- `LOW_CONFIDENCE_YOLO_WITH_RESNET_FALLBACK`: YOLOv8m produced no reliable detections but did produce candidate boxes between `0.05` and `0.65`. Each candidate box is kept in the visualization, while an expanded crop around the same box is classified by ResNet18 and explained with Grad-CAM.
- `NO_YOLO_DETECTION_RESNET_FALLBACK`: YOLOv8m returned no boxes. The whole image is classified by ResNet18 and explained with Grad-CAM.

Every run returns a structured dictionary with YOLO detections, optional ResNet18 classification, optional Grad-CAM path, final summary, and disclaimer.

## 7. Project Files

```text
facial-skin-condition-detection/
  README.md
  requirements.txt
  server.py
  Facial-Skin-Condition-Detection/
  weights/
  assets/
  outputs/
  src/
  tests/
```

Important modules:

- `server.py`: FastAPI backend that serves the frontend and exposes `/api/analyze`.
- `src/yolo_detector.py`: Ultralytics YOLOv8m wrapper.
- `src/resnet_classifier.py`: TorchVision ResNet18 classifier wrapper.
- `src/gradcam_explainer.py`: custom Grad-CAM implementation for ResNet18.
- `src/decision_engine.py`: hybrid decision pipeline.
- `src/visualization.py`: bounding-box and candidate-region drawing.
- `src/labels.py`: class lists, semantic mapping, and broad-category helpers.

## 8. Tests

```bash
pytest
```

The tests cover label mapping, preprocessing, and the three decision-engine branches with mocked model outputs.

## 9. Disclaimer

This project is for educational demonstration only and should not be used as a medical diagnosis tool.
