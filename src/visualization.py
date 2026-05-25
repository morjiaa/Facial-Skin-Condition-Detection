from typing import Iterable

import cv2
import numpy as np
from PIL import Image


def draw_detections(image: Image.Image, detections: Iterable[dict], candidate: bool = False) -> Image.Image:
    canvas = np.array(image.convert("RGB")).copy()
    color = (255, 175, 0) if candidate else (0, 180, 80)
    thickness = 2 if candidate else 3

    for detection in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in detection["box_xyxy"]]
        if candidate:
            _draw_dashed_rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        else:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        label = f'{detection["class_name"]} {detection["confidence"]:.2f} | {detection["broad_category"]}'
        _draw_label(canvas, label, x1, y1, color)

    return Image.fromarray(canvas)


def _draw_label(canvas, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    y = max(y, text_h + baseline + 4)
    cv2.rectangle(canvas, (x, y - text_h - baseline - 6), (x + text_w + 6, y + 2), color, -1)
    cv2.putText(canvas, text, (x + 3, y - baseline - 2), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def _draw_dashed_rectangle(canvas, pt1, pt2, color, thickness, dash_length=12):
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, dash_length * 2):
        cv2.line(canvas, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        cv2.line(canvas, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
    for y in range(y1, y2, dash_length * 2):
        cv2.line(canvas, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        cv2.line(canvas, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
