from typing import Optional


YOLO_CLASSES = [
    "Acne",
    "Blackhead",
    "Conglobata",
    "Crystalline",
    "Cystic",
    "Flat Wart",
    "Folliculitis",
    "Keloid",
    "Milium",
    "Papular",
    "Purulent",
    "Scars",
    "Sebo-crystan-conglo",
    "Syringoma",
    "Whitehead",
]

RESNET_DATASET_1_CLASSES = [
    "Acne",
    "Carcinoma",
    "Eczema",
    "Keratosis",
    "Milia",
    "Rosacea",
]

RESNET_DATASET_2_CLASSES = [
    "Blackheads",
    "Cyst",
    "Papules",
    "Pustules",
    "Whiteheads",
]

RESNET_CLASSES = [
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

YOLO_TO_RESNET = {
    "Acne": "Acne",
    "Blackhead": "Blackheads",
    "Whitehead": "Whiteheads",
    "Papular": "Papules",
    "Purulent": "Pustules",
    "Cystic": "Cyst",
    "Milium": "Milia",
}

YOLO_ONLY_CLASSES = [
    "Conglobata",
    "Crystalline",
    "Flat Wart",
    "Folliculitis",
    "Keloid",
    "Scars",
    "Sebo-crystan-conglo",
    "Syringoma",
]

RESNET_ONLY_CLASSES = [
    "Carcinoma",
    "Eczema",
    "Keratosis",
    "Rosacea",
]

BROAD_CATEGORY_MAP = {
    "Acne": "Acne-spectrum lesions",
    "Blackhead": "Acne-spectrum lesions",
    "Blackheads": "Acne-spectrum lesions",
    "Whitehead": "Acne-spectrum lesions",
    "Whiteheads": "Acne-spectrum lesions",
    "Papular": "Acne-spectrum lesions",
    "Papules": "Acne-spectrum lesions",
    "Purulent": "Acne-spectrum lesions",
    "Pustules": "Acne-spectrum lesions",
    "Cystic": "Acne-spectrum lesions",
    "Cyst": "Acne-spectrum lesions",
    "Conglobata": "Acne-spectrum lesions",
    "Milium": "Milia / keratin cyst",
    "Milia": "Milia / keratin cyst",
    "Scars": "Scar-related changes",
    "Keloid": "Scar-related changes",
    "Folliculitis": "Follicular inflammation",
    "Flat Wart": "Benign lesions / growths",
    "Syringoma": "Benign lesions / growths",
    "Eczema": "Other inflammatory skin conditions",
    "Rosacea": "Other inflammatory skin conditions",
    "Keratosis": "Keratosis / cancer-related",
    "Carcinoma": "Keratosis / cancer-related",
    "Crystalline": "YOLO-specific unclear label",
    "Sebo-crystan-conglo": "YOLO-specific unclear label",
}


def get_broad_category(label: str) -> str:
    return BROAD_CATEGORY_MAP.get(label, "Unmapped category")


def is_yolo_only(label: str) -> bool:
    return label in YOLO_ONLY_CLASSES


def is_resnet_only(label: str) -> bool:
    return label in RESNET_ONLY_CLASSES


def map_yolo_to_resnet(label: str) -> Optional[str]:
    return YOLO_TO_RESNET.get(label)
