from PIL import Image

from src.preprocessing import pil_to_cv2_bgr, prepare_resnet_tensor
from src.utils import crop_image_using_box


def test_prepare_resnet_tensor_shape():
    image = Image.new("RGB", (320, 240), color="white")
    tensor = prepare_resnet_tensor(image)
    assert tuple(tensor.shape) == (1, 3, 224, 224)


def test_pil_to_cv2_bgr_shape():
    image = Image.new("RGB", (10, 20), color="white")
    array = pil_to_cv2_bgr(image)
    assert array.shape == (20, 10, 3)


def test_crop_image_using_box_clamps_bounds():
    image = Image.new("RGB", (100, 80), color="white")
    crop = crop_image_using_box(image, [-10, -5, 40, 20])
    assert crop.size == (40, 20)
