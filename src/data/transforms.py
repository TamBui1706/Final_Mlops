"""Data augmentation transforms for rice disease classification."""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """
    Get training data augmentation pipeline.

    Args:
        image_size: Target image size

    Returns:
        Albumentations Compose object
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.HueSaturationValue(
                        hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0
                    ),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=3, p=1.0),
                    A.GaussNoise(p=1.0),
                ],
                p=0.3,
            ),
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                p=0.5,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(image_size // 16, image_size // 8),
                hole_width_range=(image_size // 16, image_size // 8),
                p=0.3,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """
    Get validation data transforms.

    Args:
        image_size: Target image size

    Returns:
        Albumentations Compose object
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def get_test_time_augmentation(image_size: int = 224) -> list:
    """
    Get test time augmentation transforms.

    Args:
        image_size: Target image size

    Returns:
        List of Albumentations Compose objects
    """
    return [
        A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        ),
        A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        ),
        A.Compose(
            [
                A.Resize(image_size, image_size),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        ),
    ]
