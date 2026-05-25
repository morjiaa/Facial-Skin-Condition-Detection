from . import config
from .labels import get_broad_category
from .utils import crop_image_using_box


class AcneScanDecisionEngine:
    def __init__(self, yolo_detector, classifier, gradcam_explainer):
        self.yolo_detector = yolo_detector
        self.resnet_classifier = classifier
        self.gradcam_explainer = gradcam_explainer
        self.pipeline_name = "AcneScan"

    def run(self, image):
        yolo_detections = self.yolo_detector.predict(image)
        candidate_results = []

        yolo_payload = {
            "used": True,
            "all_detections": yolo_detections,
            "reliable_detections": [],
            "low_confidence_detections": yolo_detections,
        }

        for detection in yolo_detections:
            crop = crop_image_using_box(
                image,
                detection["box_xyxy"],
                padding_ratio=config.ACNESCAN_CROP_PADDING,
            )
            classifier_result = self.resnet_classifier.predict(crop, top_k=config.TOP_K)
            gradcam = self._try_gradcam(crop, classifier_result["predicted_class"])
            public_classifier_result = {
                "predicted_class": classifier_result["predicted_class"],
                "probability": classifier_result["probability"],
                "top_k": classifier_result["top_k"],
                "broad_category": classifier_result["broad_category"],
            }
            candidate_results.append(
                {
                    "yolo_class_name": detection["class_name"],
                    "yolo_confidence": detection["confidence"],
                    "box_xyxy": detection["box_xyxy"],
                    "resnet_result": public_classifier_result,
                    "gradcam": {
                        "path": gradcam.get("path") if gradcam else None,
                        "error": gradcam.get("error") if gradcam else None,
                        "target_class": gradcam.get("target_class") if gradcam else None,
                    },
                    "explanation": "The region was located by YOLO and classified by EfficientNet-B0.",
                }
            )

        if candidate_results:
            yolo_payload["candidate_classifications"] = candidate_results
            primary = candidate_results[0]["resnet_result"]
            gradcam = candidate_results[0]["gradcam"]
            return self._result(
                result_type="ACNESCAN_YOLO_EFFICIENTNET_REGIONS",
                model_source="YOLOv8 detection + EfficientNet-B0",
                visualization_type="Detection boxes + Grad-CAM crop",
                yolo_payload=yolo_payload,
                classifier_result=primary,
                gradcam=gradcam,
                input_type="crop",
                explanation="Visible concern regions were located first, then each region was classified by the acne subtype classifier.",
                status="AcneScan path selected: YOLO boxes with EfficientNet-B0 region classification.",
            )

        classifier_result = self.resnet_classifier.predict(image, top_k=config.TOP_K)
        gradcam = self._try_gradcam(image, classifier_result["predicted_class"])
        return self._result(
            result_type="ACNESCAN_NO_LOCAL_BOX_CLASSIFICATION",
            model_source="EfficientNet-B0 fallback",
            visualization_type="Image-level classification only",
            yolo_payload=yolo_payload,
            classifier_result=classifier_result,
            gradcam=gradcam,
            input_type="whole_image",
            explanation="No local concern boxes were found, so the whole image was summarized by the classifier.",
            status="No YOLO boxes found; EfficientNet-B0 used on the whole image.",
        )

    def _try_gradcam(self, image, target_class):
        try:
            return self.gradcam_explainer.generate(image, target_class=target_class)
        except Exception as exc:
            return {"path": None, "overlay": None, "error": str(exc)}

    def _result(
        self,
        result_type,
        model_source,
        visualization_type,
        yolo_payload,
        classifier_result,
        gradcam,
        input_type,
        explanation,
        status,
    ):
        gradcam_path = gradcam.get("path") if gradcam else None
        predicted_class = classifier_result["predicted_class"]
        return {
            "result_type": result_type,
            "model_source": model_source,
            "visualization_type": visualization_type,
            "yolo": yolo_payload,
            "resnet": {
                "used": True,
                "input_type": input_type,
                "predicted_class": predicted_class,
                "probability": classifier_result["probability"],
                "top_k": classifier_result["top_k"],
                "broad_category": classifier_result["broad_category"],
                "gradcam_available": bool(gradcam_path),
                "gradcam_image_path": gradcam_path,
                "gradcam_error": gradcam.get("error") if gradcam else None,
            },
            "final_summary": {
                "primary_result": predicted_class,
                "broad_category": get_broad_category(predicted_class),
                "confidence_or_probability": classifier_result["probability"],
                "explanation": explanation,
            },
            "disclaimer": config.DISCLAIMER,
            "status_messages": [status],
        }
