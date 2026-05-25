# AI-Based Facial Skin Condition Detection

A FastAPI web application for a deep learning course project that performs confidence-aware facial skin concern detection and visual explanation.

The system uses YOLOv8 for localized skin concern detection, ResNet18 for auxiliary image-level classification when detection is uncertain, and Grad-CAM to visualize regions that influenced classifier predictions. The frontend provides an upload/camera workflow, visual analysis summary, region-level explanations, and educational skin-care guidance.

> For educational demonstration only. This project is not a medical diagnosis tool.

## Features

- Upload an image or capture one with the browser camera.
- Run YOLO-based skin concern detection on the submitted facial image.
- Display detected concern regions with bounding boxes and class labels.
- Use ResNet18 fallback classification when YOLO detections are uncertain or absent.
- Generate Grad-CAM explanation images for auxiliary classification results.
- Present user-friendly visual summaries, concern counts, and selected-region explanations.
- Keep model outputs structured for easy debugging and extension.

## System Architecture

The application follows a hybrid inference pipeline:

```text
Input facial image
    -> YOLOv8 detection
    -> Confidence-based decision engine
    -> High-confidence YOLO result
       or ResNet18 fallback classification
    -> Optional Grad-CAM explanation
    -> Frontend visual report
```

YOLO is the only component that produces bounding boxes. ResNet18 is used as an auxiliary classifier and does not produce boxes. Grad-CAM highlights image regions that influenced ResNet18 classification and should not be interpreted as precise lesion localization.

## Label Spaces

The project uses separate label spaces for detection and classification rather than forcing all labels into one shared class list.

YOLO detection classes:

```text
Acne
Blackhead
Conglobata
Crystalline
Cystic
Flat Wart
Folliculitis
Keloid
Milium
Papular
Purulent
Scars
Sebo-crystan-conglo
Syringoma
Whitehead
```

Default ResNet18 classification classes:

```text
Acne
Carcinoma
Eczema
Keratosis
Milia
Rosacea
Blackheads
Cyst
Papules
Pustules
Whiteheads
```

If your trained ResNet18 model used a different class order, update `RESNET_CLASSES` in `src/labels.py` before running inference.

## Decision Logic

The decision engine uses three main paths:

- `YOLO_HIGH_CONFIDENCE_DETECTION`: YOLO returns at least one detection with confidence greater than or equal to `0.65`. Reliable boxes are displayed and ResNet18 is not used.
- `LOW_CONFIDENCE_YOLO_WITH_RESNET_FALLBACK`: YOLO returns candidate boxes below the reliable threshold. Candidate boxes remain visible, while expanded crops are classified by ResNet18 and explained with Grad-CAM.
- `NO_YOLO_DETECTION_RESNET_FALLBACK`: YOLO returns no boxes. The whole image is classified by ResNet18, and no bounding box is drawn.

## Project Structure

```text
.
├── server.py
├── requirements.txt
├── README.md
├── Facial-Skin-Condition-Detection/
│   ├── index.html
│   ├── script.js
│   ├── styles.css
│   └── hero-preview.jpg
├── src/
│   ├── config.py
│   ├── labels.py
│   ├── yolo_detector.py
│   ├── resnet_classifier.py
│   ├── gradcam_explainer.py
│   ├── decision_engine.py
│   ├── visualization.py
│   └── utils.py
├── tests/
├── weights/
└── outputs/
```

## Model Weights

Model weights are not included in this repository because trained `.pt` and `.pth` files can be large.

Place your trained weights in:

```text
weights/best.pt
weights/best_model.pt
```

The default paths can be changed in `src/config.py`:

```python
YOLO_MODEL_PATH = WEIGHTS_DIR / "best.pt"
RESNET_MODEL_PATH = WEIGHTS_DIR / "best_model.pt"
```

If the model files are missing, the backend can still run in demo mode for frontend preview, but predictions will not represent real model inference.

## Installation

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
uvicorn server:app --reload
```

Open the frontend:

```text
http://127.0.0.1:8000
```

## API

The frontend sends images to:

```text
POST /api/analyze
```

The backend returns a structured inference result containing:

- YOLO detections
- ResNet18 fallback output when used
- Grad-CAM image path when generated
- frontend visualization metadata
- final visual summary
- educational disclaimer

Generated upload, detection, and Grad-CAM files are written under `outputs/`.

## Configuration

Key inference settings are defined in `src/config.py`:

```python
RELIABLE_YOLO_CONFIDENCE = 0.65
RAW_YOLO_CONFIDENCE = 0.05
YOLO_IOU_THRESHOLD = 0.45
RESNET_IMAGE_SIZE = 224
TOP_K = 3
LOW_CONFIDENCE_CROP_PADDING = 0.35
```

YOLO inference runs with the raw confidence threshold so uncertain candidate boxes remain available for fallback classification.

## Tests

Run:

```bash
pytest
```

The tests cover label mapping, preprocessing, and decision-engine behavior using mocked model outputs.

## Disclaimer

This project is for educational demonstration only. The results are AI-generated visual predictions and should not be used as medical diagnosis or treatment advice. If you are concerned about a real skin condition, consult a qualified healthcare professional.
