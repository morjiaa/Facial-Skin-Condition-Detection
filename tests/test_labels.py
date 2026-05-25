from src.labels import (
    get_broad_category,
    is_resnet_only,
    is_yolo_only,
    map_yolo_to_resnet,
)


def test_map_yolo_to_resnet_blackhead():
    assert map_yolo_to_resnet("Blackhead") == "Blackheads"


def test_get_broad_category_whiteheads():
    assert get_broad_category("Whiteheads") == "Acne-spectrum lesions"


def test_get_broad_category_carcinoma():
    assert get_broad_category("Carcinoma") == "Keratosis / cancer-related"


def test_is_yolo_only_folliculitis():
    assert is_yolo_only("Folliculitis") is True


def test_is_resnet_only_rosacea():
    assert is_resnet_only("Rosacea") is True
