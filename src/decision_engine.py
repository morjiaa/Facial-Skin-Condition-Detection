from . import config
from .labels import get_broad_category, is_yolo_only, map_yolo_to_resnet
from .utils import crop_image_using_box


class HybridDecisionEngine:
    def __init__(self, yolo_detector, resnet_classifier, gradcam_explainer):
        self.yolo_detector = yolo_detector
        self.resnet_classifier = resnet_classifier
        self.gradcam_explainer = gradcam_explainer

    def run(self, image):
        yolo_detections = self.yolo_detector.predict(image)
        reliable = [d for d in yolo_detections if d["confidence"] >= config.RELIABLE_YOLO_CONFIDENCE]
        low_conf = [
            d
            for d in yolo_detections
            if config.RAW_YOLO_CONFIDENCE <= d["confidence"] < config.RELIABLE_YOLO_CONFIDENCE
        ]

        yolo_payload = {
            "used": True,
            "all_detections": yolo_detections,
            "reliable_detections": reliable,
            "low_confidence_detections": low_conf,
        }

        if reliable:
            best = reliable[0]
            return {
                "result_type": "YOLO_HIGH_CONFIDENCE_DETECTION",
                "model_source": "YOLOv8m",
                "visualization_type": "Bounding boxes",
                "yolo": yolo_payload,
                "resnet": {"used": False, "input_type": None, "gradcam_available": False},
                "final_summary": {
                    "primary_result": best["class_name"],
                    "broad_category": best["broad_category"],
                    "confidence_or_probability": best["confidence"],
                    "explanation": "YOLOv8m produced reliable detection results, so ResNet18 fallback was not used.",
                },
                "disclaimer": config.DISCLAIMER,
                "status_messages": ["High-confidence YOLOv8m detection path selected."],
            }

        if low_conf:
            candidate_results = []
            for candidate in low_conf:
                crop = crop_image_using_box(
                    image,
                    candidate["box_xyxy"],
                    padding_ratio=config.LOW_CONFIDENCE_CROP_PADDING,
                )
                resnet_result = self.resnet_classifier.predict(crop, top_k=config.TOP_K)
                gradcam = self._try_gradcam(crop, resnet_result["predicted_class"])
                public_resnet_result = {
                    "predicted_class": resnet_result["predicted_class"],
                    "probability": resnet_result["probability"],
                    "top_k": resnet_result["top_k"],
                    "broad_category": resnet_result["broad_category"],
                }
                public_gradcam = {
                    "path": gradcam.get("path") if gradcam else None,
                    "error": gradcam.get("error") if gradcam else None,
                    "target_class": gradcam.get("target_class") if gradcam else None,
                }
                candidate_results.append(
                    {
                        "yolo_class_name": candidate["class_name"],
                        "yolo_confidence": candidate["confidence"],
                        "box_xyxy": candidate["box_xyxy"],
                        "resnet_result": public_resnet_result,
                        "gradcam": public_gradcam,
                        "explanation": self._low_confidence_explanation(
                            candidate["class_name"],
                            resnet_result["predicted_class"],
                        ),
                    }
                )

            primary_candidate_result = candidate_results[0]
            yolo_payload["candidate_classifications"] = candidate_results
            resnet_result = primary_candidate_result["resnet_result"]
            gradcam = primary_candidate_result["gradcam"]
            return self._fallback_result(
                result_type="LOW_CONFIDENCE_YOLO_WITH_RESNET_FALLBACK",
                model_source="YOLOv8m + ResNet18",
                visualization_type="Candidate box + Grad-CAM heatmap",
                yolo_payload=yolo_payload,
                resnet_result=resnet_result,
                gradcam=gradcam,
                input_type="crop",
                explanation=primary_candidate_result["explanation"],
                status="Low-confidence YOLOv8m candidate region(s) found; ResNet18 fallback used on expanded candidate crop(s).",
            )

        resnet_result = self.resnet_classifier.predict(image, top_k=config.TOP_K)
        gradcam = self._try_gradcam(image, resnet_result["predicted_class"])
        return self._fallback_result(
            result_type="NO_YOLO_DETECTION_RESNET_FALLBACK",
            model_source="ResNet18 fallback",
            visualization_type="Grad-CAM heatmap only",
            yolo_payload=yolo_payload,
            resnet_result=resnet_result,
            gradcam=gradcam,
            input_type="whole_image",
            explanation="YOLOv8m returned no candidate boxes, so ResNet18 was used for image-level auxiliary classification.",
            status="No YOLOv8m detections found; ResNet18 fallback used on the whole image.",
        )

    def _try_gradcam(self, image, target_class):
        try:
            return self.gradcam_explainer.generate(image, target_class=target_class)
        except Exception as exc:
            return {"path": None, "overlay": None, "error": str(exc)}

    def _low_confidence_explanation(self, yolo_label, resnet_label):
        mapped = map_yolo_to_resnet(yolo_label)
        if mapped and mapped == resnet_label:
            return "Auxiliary classifier supports the YOLO candidate category."
        if is_yolo_only(yolo_label):
            return "This YOLO class is not directly available in the ResNet18 label space. ResNet18 output is shown as auxiliary reference only."
        if mapped:
            return "YOLO candidate and ResNet18 auxiliary classification do not exactly match; review both outputs as course-demo evidence."
        return "ResNet18 output is shown as auxiliary image-level reference only."

    def _fallback_result(
        self,
        result_type,
        model_source,
        visualization_type,
        yolo_payload,
        resnet_result,
        gradcam,
        input_type,
        explanation,
        status,
    ):
        gradcam_path = gradcam.get("path") if gradcam else None
        return {
            "result_type": result_type,
            "model_source": model_source,
            "visualization_type": visualization_type,
            "yolo": yolo_payload,
            "resnet": {
                "used": True,
                "input_type": input_type,
                "predicted_class": resnet_result["predicted_class"],
                "probability": resnet_result["probability"],
                "top_k": resnet_result["top_k"],
                "broad_category": resnet_result["broad_category"],
                "gradcam_available": bool(gradcam_path),
                "gradcam_image_path": gradcam_path,
                "gradcam_error": gradcam.get("error") if gradcam else None,
            },
            "final_summary": {
                "primary_result": resnet_result["predicted_class"],
                "broad_category": get_broad_category(resnet_result["predicted_class"]),
                "confidence_or_probability": resnet_result["probability"],
                "explanation": explanation,
            },
            "disclaimer": config.DISCLAIMER,
            "status_messages": [status],
        }
