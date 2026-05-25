import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

from . import config


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or f"upload_{timestamp()}"


def create_output_dirs() -> None:
    for path in [
        config.OUTPUT_DIR,
        config.OUTPUT_DIR / "yolo",
        config.OUTPUT_DIR / "gradcam",
        config.OUTPUT_DIR / "uploads",
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_uploaded_image(image: Image.Image, original_name: str = "uploaded.png") -> Path:
    create_output_dirs()
    path = config.OUTPUT_DIR / "uploads" / f"{timestamp()}_{safe_filename(original_name)}"
    image.save(path)
    return path


def crop_image_using_box(image: Image.Image, box_xyxy: Sequence[float], padding_ratio: float = 0.0) -> Image.Image:
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
    return image.crop((x1, y1, x2, y2))


def format_probability(value: float) -> str:
    return f"{value * 100:.1f}%"


def ensure_list(values: Iterable) -> list:
    return list(values) if values is not None else []
