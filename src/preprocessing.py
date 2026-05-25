from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from .config import RESNET_IMAGE_SIZE


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_uploaded_image(uploaded_file) -> Image.Image:
    data = uploaded_file.read()
    image = Image.open(BytesIO(data)).convert("RGB")
    return image


def pil_to_cv2_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def pil_to_cv2_rgb(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def get_resnet_transform(image_size: int = RESNET_IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def prepare_resnet_tensor(image: Image.Image, image_size: int = RESNET_IMAGE_SIZE):
    transform = get_resnet_transform(image_size)
    return transform(image.convert("RGB")).unsqueeze(0)
