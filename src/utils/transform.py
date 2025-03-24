from typing import Literal

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

# Different between ToTensorV2 and ToTensor
# ToTensorV2: Converts image and mask to torch.Tensor.
# ToTensor: Convert image and mask to torch.Tensor but scale the image pixel values to the range [0, 1]. This one is deprecated.


class Transform:
    def __init__(self, mode: Literal["train", "val"]):
        if mode == "train":
            self.aug = A.Compose(
                [
                    A.Resize(150, 150, interpolation=cv2.INTER_LINEAR),
                    A.VerticalFlip(p=0.3),
                    A.HorizontalFlip(p=0.3),
                    A.RGBShift(p=0.3),
                    A.RandomBrightnessContrast(p=0.3),
                    A.GaussNoise(p=0.3),
                    A.Rotate(limit=30, p=0.3),
                    A.OneOf(
                        [
                            A.Blur(blur_limit=3, p=0.3),
                            A.GaussianBlur(blur_limit=3, p=0.3),
                        ]
                    ),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
        else:
            self.aug = A.Compose(
                [
                    A.Resize(150, 150, interpolation=cv2.INTER_LINEAR),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.aug(image=image)["image"]

        # Tips: Another params when using albumentations is "mask" for segmentation task
