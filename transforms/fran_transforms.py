from typing import Callable, List, Tuple

import torch
from torchvision.transforms import CenterCrop, Compose, ToTensor, Normalize, \
    RandomCrop, ColorJitter, RandomRotation


def data_transforms(
    crop_size: int,
    norm_mean: List[float],
    norm_std: List[float],
    jitter_brightness: float,
    jitter_contrast: float,
    jitter_saturation: float,
    jitter_hue: float,
    random_angle: float,
) -> Tuple[Callable, Callable]:
    """Creates the data transforms for FRAN training and validation."""

    train_tfms = [
        RandomCrop(crop_size),
        ToTensor(),
        Normalize(mean=norm_mean, std=norm_std),
        ColorJitter(
            brightness=jitter_brightness,
            contrast=jitter_contrast,
            saturation=jitter_saturation,
            hue=jitter_hue,
        )
    ]

    if random_angle != 0:
        train_tfms.append(
            RandomRotation(random_angle)
        )

    tfm_train = Compose(train_tfms)

    tfm_val = Compose([
        CenterCrop(crop_size),
        ToTensor(),
        Normalize(mean=norm_mean, std=norm_std)
    ])

    return tfm_train, tfm_val
