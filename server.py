from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src import config
from src.acnescan_engine import AcneScanDecisionEngine
from src.decision_engine import HybridDecisionEngine
from src.efficientnet_classifier import EfficientNetAcneClassifier
from src.flexible_gradcam_explainer import FlexibleGradCAMExplainer
from src.gradcam_explainer import GradCAMExplainer
from src.labels import get_broad_category
from src.preprocessing import load_uploaded_image
from src.resnet_classifier import ResNetClassifier
from src.utils import create_output_dirs, crop_image_using_box, save_uploaded_image, timestamp
from src.visualization import draw_detections
from src.yolo_detector import YOLODetector


FRONTEND_DIR = config.BASE_DIR / "Facial-Skin-Condition-Detection"

app = FastAPI(title="AI-Based Facial Skin Condition Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.endswith((".html", ".js", ".css")):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


def _missing_weight_paths() -> list[str]:
    if (
        config.USE_CLASSMATE_ACNESCAN_PIPELINE
        and Path(config.CLASSMATE_YOLO_MODEL_PATH).exists()
        and Path(config.CLASSMATE_CLASSIFIER_MODEL_PATH).exists()
    ):
        return []
    return [str(path) for path in [config.YOLO_MODEL_PATH, config.RESNET_MODEL_PATH] if not Path(path).exists()]


class DemoYOLODetector:
    def predict(self, image):
        width, height = image.size
        x1 = width * 0.28
        y1 = height * 0.25
        x2 = width * 0.58
        y2 = height * 0.58
        return [
            {
                "class_name": "Papular",
                "confidence": 0.52,
                "box_xyxy": [x1, y1, x2, y2],
                "broad_category": get_broad_category("Papular"),
            },
            {
                "class_name": "Whitehead",
                "confidence": 0.41,
                "box_xyxy": [width * 0.56, height * 0.34, width * 0.78, height * 0.55],
                "broad_category": get_broad_category("Whitehead"),
            },
        ]


class DemoResNetClassifier:
    class_names = ["Papules", "Whiteheads", "Blackheads"]

    def predict(self, _image, top_k=3):
        predictions = [
            {"class_name": "Papules", "probability": 0.73, "broad_category": get_broad_category("Papules")},
            {"class_name": "Whiteheads", "probability": 0.18, "broad_category": get_broad_category("Whiteheads")},
            {"class_name": "Blackheads", "probability": 0.09, "broad_category": get_broad_category("Blackheads")},
        ]
        return {
            "predicted_class": predictions[0]["class_name"],
            "probability": predictions[0]["probability"],
            "top_k": predictions[:top_k],
            "broad_category": predictions[0]["broad_category"],
            "logits": None,
            "input_tensor": None,
        }


class DemoGradCAMExplainer:
    def generate(self, image, target_class=None):
        create_output_dirs()
        rgb = np.array(image.convert("RGB"))
        height, width = rgb.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width]
        center_x = width * 0.52
        center_y = height * 0.48
        sigma = max(width, height) * 0.23
        heatmap = np.exp(-(((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma**2)))
        heatmap = heatmap / heatmap.max()

        colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(rgb, 0.58, colored, 0.42, 0)

        output_path = config.OUTPUT_DIR / "gradcam" / f"demo_gradcam_{timestamp()}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return {
            "heatmap": heatmap,
            "overlay": overlay,
            "path": str(output_path),
            "target_class": target_class or "Papules",
        }


@lru_cache(maxsize=1)
def get_engine() -> HybridDecisionEngine:
    if (
        config.USE_CLASSMATE_ACNESCAN_PIPELINE
        and Path(config.CLASSMATE_YOLO_MODEL_PATH).exists()
        and Path(config.CLASSMATE_CLASSIFIER_MODEL_PATH).exists()
    ):
        yolo = YOLODetector(
            model_path=config.CLASSMATE_YOLO_MODEL_PATH,
            conf=config.ACNESCAN_BOX_CONFIDENCE,
            iou=config.YOLO_IOU_THRESHOLD,
            device=config.DEVICE,
        )
        classifier = EfficientNetAcneClassifier(
            model_path=config.CLASSMATE_CLASSIFIER_MODEL_PATH,
            device=config.DEVICE,
        )
        gradcam = FlexibleGradCAMExplainer(classifier)
        engine = AcneScanDecisionEngine(yolo, classifier, gradcam)
        engine.demo_mode = False
        engine.missing_weight_paths = []
        return engine

    missing = _missing_weight_paths()
    yolo_missing = not Path(config.YOLO_MODEL_PATH).exists()
    resnet_missing = not Path(config.RESNET_MODEL_PATH).exists()

    if yolo_missing:
        yolo = DemoYOLODetector()
    else:
        yolo = YOLODetector(
            model_path=config.YOLO_MODEL_PATH,
            conf=config.RAW_YOLO_CONFIDENCE,
            iou=config.YOLO_IOU_THRESHOLD,
            device=config.DEVICE,
        )

    if resnet_missing:
        resnet = DemoResNetClassifier()
        gradcam = DemoGradCAMExplainer()
    else:
        resnet = ResNetClassifier(
            model_path=config.RESNET_MODEL_PATH,
            class_names=None,
            device=config.DEVICE,
        )
        gradcam = GradCAMExplainer(resnet)

    engine = HybridDecisionEngine(yolo, resnet, gradcam)
    engine.demo_mode = bool(missing)
    engine.missing_weight_paths = missing
    return engine


def _public_output_url(path: Optional[Union[str, Path]]) -> Optional[str]:
    if not path:
        return None
    output_path = Path(path)
    try:
        relative = output_path.resolve().relative_to(config.OUTPUT_DIR.resolve())
    except ValueError:
        return None
    return "/outputs/" + relative.as_posix()


def _save_yolo_visualization(image, result: Dict) -> Optional[Path]:
    yolo = result["yolo"]
    detections = []
    candidate = False

    if result["result_type"] == "YOLO_HIGH_CONFIDENCE_DETECTION":
        detections = yolo["reliable_detections"]
    elif result["result_type"] in {"LOW_CONFIDENCE_YOLO_WITH_RESNET_FALLBACK", "ACNESCAN_YOLO_EFFICIENTNET_REGIONS"}:
        detections = yolo["low_confidence_detections"]
        candidate = True

    if not detections:
        return None

    create_output_dirs()
    visualized = draw_detections(image, detections, candidate=candidate)
    output_path = config.OUTPUT_DIR / "yolo" / f"yolo_{timestamp()}.png"
    visualized.save(output_path)
    return output_path


def _save_crop(image, box_xyxy, prefix: str, padding_ratio: float = 0.0) -> Path:
    create_output_dirs()
    crop = crop_image_using_box(image, box_xyxy, padding_ratio=padding_ratio)
    output_path = config.OUTPUT_DIR / "uploads" / f"{prefix}_{timestamp()}.png"
    crop.save(output_path)
    return output_path


def _expanded_box(image, box_xyxy, padding_ratio: float = 0.0) -> tuple[int, int, int, int]:
    width, height = image.size
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    if padding_ratio > 0:
        box_width = x2 - x1
        box_height = y2 - y1
        pad_x = box_width * padding_ratio
        pad_y = box_height * padding_ratio
        x1 -= pad_x
        y1 -= pad_y
        x2 += pad_x
        y2 += pad_y
    x1, y1, x2, y2 = [int(round(v)) for v in [x1, y1, x2, y2]]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def _context_box_around_target(image, box_xyxy, area_ratio: float = 0.25) -> tuple[int, int, int, int]:
    width, height = image.size
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    scale = area_ratio**0.5
    crop_width = max(1, int(round(width * scale)))
    crop_height = max(1, int(round(height * scale)))
    left = int(round(center_x - crop_width / 2))
    top = int(round(center_y - crop_height / 2))
    left = max(0, min(left, width - crop_width))
    top = max(0, min(top, height - crop_height))
    return left, top, left + crop_width, top + crop_height


def _save_context_gradcam(image, box_xyxy, gradcam_path, prefix: str, padding_ratio: float = 0.0) -> Optional[Path]:
    if not gradcam_path:
        return None
    source_path = Path(gradcam_path)
    if not source_path.exists():
        return None

    create_output_dirs()
    x1, y1, x2, y2 = _expanded_box(image, box_xyxy, padding_ratio=padding_ratio)
    cx1, cy1, cx2, cy2 = _context_box_around_target(image, box_xyxy, area_ratio=0.25)
    context = image.convert("RGB").crop((cx1, cy1, cx2, cy2))

    from PIL import Image

    overlay = Image.open(source_path).convert("RGB").resize((x2 - x1, y2 - y1))
    ix1 = max(x1, cx1)
    iy1 = max(y1, cy1)
    ix2 = min(x2, cx2)
    iy2 = min(y2, cy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None

    overlay_crop = overlay.crop((ix1 - x1, iy1 - y1, ix2 - x1, iy2 - y1))
    context.paste(overlay_crop, (ix1 - cx1, iy1 - cy1))

    output_path = config.OUTPUT_DIR / "gradcam" / f"{prefix}_{timestamp()}.png"
    context.save(output_path)
    return output_path


def _label_counts(detections: Iterable[Dict], fallback_label: Optional[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for detection in detections:
        label = detection["class_name"]
        counts[label] = counts.get(label, 0) + 1
    if not counts and fallback_label:
        counts[fallback_label] = 1
    return counts


def _label_counts_from_labels(labels: Iterable[str], fallback_label: Optional[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label in labels:
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    if not counts and fallback_label:
        counts[fallback_label] = 1
    return counts


def _severity(label_counts: Dict[str, int], primary_label: str) -> str:
    total = sum(label_counts.values())
    high_attention = {"Cystic", "Cyst", "Purulent", "Pustules", "Carcinoma", "Keratosis"}
    if primary_label in high_attention or any(label in high_attention for label in label_counts):
        return "Higher attention"
    if total >= 6:
        return "Moderate"
    if total >= 1:
        return "Mild"
    return "No clear activity"


def _main_concern_group(label_counts: Dict[str, int], primary_label: str) -> str:
    if label_counts:
        return max(label_counts.items(), key=lambda item: item[1])[0]
    return primary_label or "Unclear"


def _consumer_summary(result: Dict, severity: str, label_counts: Dict[str, int], primary_label: str) -> str:
    areas_found = sum(label_counts.values())
    broad_category = get_broad_category(primary_label)
    sensitive_labels = {"Carcinoma", "Keratosis"}

    if primary_label in sensitive_labels:
        return (
            "The image shows visual patterns that fall into a sensitive skin concern category. "
            "This result should be treated as a cautious visual flag rather than a skin type label. "
            "For real skin concerns, professional review is recommended."
        )

    if result["result_type"] == "NO_YOLO_DETECTION_RESNET_FALLBACK":
        return (
            "Your skin looks generally clear in this image, with no obvious localized concern spots detected. "
            f"There may be a subtle tendency toward {primary_label}-related visual patterns, so keep an eye on changes in texture, redness, or new spots. "
            "Maintain a gentle routine, protect your skin with sunscreen, and retake the analysis in consistent lighting if you notice changes."
        )

    if broad_category == "Acne-spectrum lesions":
        if severity in {"Low", "Mild"}:
            return (
                f"The image shows a small number of visible acne-related concern areas, mainly around {primary_label}. "
                "The overall pattern appears limited, with only a few localized spots identified."
            )
        if severity == "Moderate":
            return (
                f"The image shows multiple acne-related concern areas, with {primary_label} appearing as the main visible pattern. "
                "The distribution looks more noticeable, so the result is summarized as a moderate visual concern level."
            )
        return (
            f"The image shows visually prominent acne-related concern areas, mainly associated with {primary_label}. "
            "The result should be interpreted carefully because the visible coverage is relatively strong."
        )

    if broad_category == "Milia / keratin cyst":
        return (
            "The image shows small bump-like visual patterns that are closest to a milia or keratin-cyst appearance. "
            "These areas look localized rather than broadly spread across the face."
        )

    if broad_category == "Scar-related changes":
        return (
            "The image shows visual texture changes that are closest to scar-related patterns. "
            "The summary focuses on visible surface texture rather than active inflammation."
        )

    if broad_category == "Other inflammatory skin conditions":
        return (
            f"The overall image pattern is closest to {primary_label}, which falls under broader inflammatory-looking skin concerns. "
            "Lighting, redness, and image sharpness can strongly affect this type of visual prediction."
        )

    if areas_found:
        return (
            f"The image contains {areas_found} localized visual concern region(s), with {primary_label} as the main predicted pattern. "
            "This summary is based on visible image features only."
        )

    return (
        "The image does not show many clear localized concern regions. "
        "The result should be interpreted as a broad visual pattern summary."
    )


def _build_frontend_report(result: Dict, image_size) -> Dict:
    yolo = result["yolo"]
    summary = result["final_summary"]
    fallback_label = summary["primary_result"]

    if yolo["reliable_detections"]:
        visible_count = len(yolo["reliable_detections"])
        label_counts = _label_counts(yolo["reliable_detections"], fallback_label)
    elif yolo.get("candidate_classifications"):
        visible_count = len(yolo["candidate_classifications"])
        labels = [
            item["resnet_result"]["predicted_class"]
            for item in yolo["candidate_classifications"]
        ]
        label_counts = _label_counts_from_labels(labels, fallback_label)
    else:
        visible_count = 0
        label_counts = {}

    primary_label = _main_concern_group(label_counts, fallback_label)
    severity = _severity(label_counts, primary_label)

    return {
        "severity": severity,
        "areas_found": visible_count,
        "primary_label": primary_label,
        "broad_category": get_broad_category(primary_label),
        "main_concern_group": primary_label,
        "sensitive_category_flag": primary_label in {"Carcinoma", "Keratosis"},
        "consumer_summary": _consumer_summary(result, severity, label_counts, primary_label),
        "label_counts": label_counts,
        "image_size": {"width": image_size[0], "height": image_size[1]},
    }


def _attach_frontend_payload(result: Dict, uploaded_path: Path, yolo_path: Optional[Path], image_size) -> Dict:
    gradcam_path = result.get("resnet", {}).get("gradcam_image_path")
    gradcam_url = _public_output_url(gradcam_path)

    result["frontend_report"] = _build_frontend_report(result, image_size)
    result["visualizations"] = {
        "uploaded_image_url": _public_output_url(uploaded_path),
        "yolo_image_url": _public_output_url(yolo_path),
        "gradcam_image_url": gradcam_url,
    }

    if result.get("resnet", {}).get("used"):
        result["resnet"]["gradcam_image_url"] = gradcam_url

    return result


def _build_detail_items(image, result: Dict, uploaded_image_url: Optional[str]) -> list[Dict]:
    yolo = result["yolo"]
    resnet = result.get("resnet", {})
    gradcam_url = _public_output_url(resnet.get("gradcam_image_path"))
    items = []

    if result["result_type"] == "YOLO_HIGH_CONFIDENCE_DETECTION":
        for index, detection in enumerate(yolo["reliable_detections"]):
            crop_path = _save_crop(image, detection["box_xyxy"], f"reliable_crop_{index}")
            items.append(
                {
                    "kind": "yolo_original_crop",
                    "title": f'{detection["class_name"]} detection',
                    "display_mode": "original_crop",
                    "label": detection["class_name"],
                    "confidence": detection["confidence"],
                    "box_xyxy": detection["box_xyxy"],
                    "crop_image_url": _public_output_url(crop_path),
                    "gradcam_image_url": None,
                    "yolo_decision": f'{detection["class_name"]} · {detection["confidence"]:.0%}',
                    "resnet_decision": "Not used",
                    "explanation": "YOLOv8m produced a reliable detection, so ResNet18 fallback and Grad-CAM were not used for this region.",
                    "summary": "Original crop is shown because this region passed the reliable YOLO confidence threshold.",
                    "broad_category": detection["broad_category"],
                }
            )
        return items

    if result["result_type"] in {"LOW_CONFIDENCE_YOLO_WITH_RESNET_FALLBACK", "ACNESCAN_YOLO_EFFICIENTNET_REGIONS"}:
        candidate_classifications = yolo.get("candidate_classifications", [])
        for index, detection in enumerate(yolo["low_confidence_detections"]):
            auxiliary = candidate_classifications[index] if index < len(candidate_classifications) else None
            resnet_result = auxiliary["resnet_result"] if auxiliary else {}
            candidate_gradcam = auxiliary.get("gradcam") if auxiliary else None
            padding_ratio = (
                config.ACNESCAN_CROP_PADDING
                if result["result_type"] == "ACNESCAN_YOLO_EFFICIENTNET_REGIONS"
                else config.LOW_CONFIDENCE_CROP_PADDING
            )
            candidate_gradcam_path = candidate_gradcam.get("path") if candidate_gradcam else None
            context_gradcam_path = _save_context_gradcam(
                image,
                detection["box_xyxy"],
                candidate_gradcam_path,
                f"context_gradcam_{index}",
                padding_ratio=padding_ratio,
            )
            candidate_gradcam_url = _public_output_url(context_gradcam_path or candidate_gradcam_path)
            crop_path = _save_crop(
                image,
                detection["box_xyxy"],
                f"candidate_crop_{index}",
                padding_ratio=padding_ratio,
            )
            predicted_class = resnet_result.get("predicted_class", detection["class_name"])
            probability = resnet_result.get("probability", detection["confidence"])
            items.append(
                {
                    "kind": "yolo_candidate_gradcam",
                    "title": f"{predicted_class} visual concern",
                    "display_mode": "gradcam_crop" if candidate_gradcam_url else "original_crop",
                    "label": predicted_class,
                    "confidence": probability,
                    "yolo_label": detection["class_name"],
                    "yolo_confidence": detection["confidence"],
                    "auxiliary_label": predicted_class,
                    "auxiliary_confidence": probability,
                    "box_xyxy": detection["box_xyxy"],
                    "crop_image_url": _public_output_url(crop_path),
                    "gradcam_image_url": candidate_gradcam_url,
                    "yolo_decision": (
                        f'Region found · {detection["confidence"]:.0%}'
                        if result["result_type"] == "ACNESCAN_YOLO_EFFICIENTNET_REGIONS"
                        else f'{detection["class_name"]} candidate · {detection["confidence"]:.0%}'
                    ),
                    "resnet_decision": f"{predicted_class} · {probability:.0%}",
                    "explanation": auxiliary.get("explanation") if auxiliary else result["final_summary"]["explanation"],
                    "summary": (
                        "This region was first located by the detector. The displayed label comes from EfficientNet-B0 classification on an expanded crop around the same box."
                        if result["result_type"] == "ACNESCAN_YOLO_EFFICIENTNET_REGIONS"
                        else "This region was first located by YOLO as a low-confidence candidate. The displayed label comes from ResNet18 classification on an expanded crop around the same box."
                    ),
                    "broad_category": resnet_result.get("broad_category", get_broad_category(predicted_class)),
                }
            )
        return items

    return items


def _mark_demo_result(result: Dict, missing: list[str]) -> Dict:
    result["demo_mode"] = True
    result["missing_model_weights"] = missing
    result.setdefault("status_messages", []).insert(
        0,
        "Some model weight files are missing. Available real models are used, and missing models are mocked for UI preview.",
    )
    result["final_summary"]["explanation"] = (
        "Some model weight files are missing. "
        + result["final_summary"]["explanation"]
    )
    return result


@app.get("/api/health")
def health_check():
    missing = _missing_weight_paths()
    return {
        "status": "ok",
        "device": config.DEVICE,
        "demo_mode": bool(missing),
        "missing_model_weights": missing,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    try:
        image = load_uploaded_image(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {exc}") from exc

    uploaded_path = save_uploaded_image(image, file.filename or "uploaded.png")

    try:
        engine = get_engine()
        result = engine.run(image)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    if getattr(engine, "demo_mode", False):
        result = _mark_demo_result(result, getattr(engine, "missing_weight_paths", []))

    yolo_path = _save_yolo_visualization(image, result)
    result = _attach_frontend_payload(result, uploaded_path, yolo_path, image.size)
    result["detail_items"] = _build_detail_items(image, result, result["visualizations"]["uploaded_image_url"])
    return result


create_output_dirs()
app.mount("/outputs", StaticFiles(directory=config.OUTPUT_DIR), name="outputs")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
