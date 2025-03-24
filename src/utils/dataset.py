import os
from typing import List, Literal, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .transform import Transform


class CatDogDataset(Dataset):
    def __init__(
        self,
        split_ratio: float = 0.8,
        data_dir: str = None,
        files: list = None,
        seed: int = 42,
        mode: str = None,
        transform=None,
    ):
        self.split_ratio = split_ratio
        self.files = files
        self.seed = seed
        self.mode = mode
        self.transform = transform

        self.classes = {
            0: "Cat",
            1: "Dog",
        }

        # Split the data if data_dir is provided
        if data_dir:
            self.train_files, self.val_files = self.__split_data(data_dir=data_dir)

    def from_mode(self, mode: Literal["train", "val"]) -> "CatDogDataset":
        """Create a new dataset instance with the specified mode."""
        transform = Transform(mode=mode)

        if mode == "train":
            return CatDogDataset(
                files=self.train_files,
                mode=mode,
                transform=transform,
            )
        elif mode == "val":
            return CatDogDataset(
                files=self.val_files,
                mode=mode,
                transform=transform,
            )
        else:
            raise ValueError(f"Mode {mode} not supported")

    def __split_data(self, data_dir: str) -> Tuple[List[str], List[str]]:
        """
        Split the data into training and validation sets.

        Args:
            data_dir: Directory containing the data organized in Cat and Dog folders.

        Returns:
            A tuple of two lists:
            - The first list contains paths to training files
            - The second list contains paths to validation files
        """
        # Collect file paths
        cat_files = [
            os.path.join(data_dir, "Cat", f)
            for f in os.listdir(os.path.join(data_dir, "Cat"))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        dog_files = [
            os.path.join(data_dir, "Dog", f)
            for f in os.listdir(os.path.join(data_dir, "Dog"))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # Apply seed for reproducibility before shuffling
        if self.seed is not None:
            np.random.seed(self.seed)

        # Split the files
        cat_idx = int(len(cat_files) * self.split_ratio)
        dog_idx = int(len(dog_files) * self.split_ratio)

        cat_train, cat_val = cat_files[:cat_idx], cat_files[cat_idx:]
        dog_train, dog_val = dog_files[:dog_idx], dog_files[dog_idx:]

        train_files = [(f, 0) for f in cat_train] + [(f, 1) for f in dog_train]
        val_files = [(f, 0) for f in cat_val] + [(f, 1) for f in dog_val]

        # Shuffle the files
        np.random.shuffle(train_files)
        np.random.shuffle(val_files)

        return train_files, val_files

    def __len__(self):
        return len(self.files) if self.files is not None else 0

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Return a sample from the dataset.

        Args:
            idx: Index of the sample to return

        Returns:
            A tuple of (image, label)
        """
        file_path, label = self.files[idx]

        # Load image
        image = Image.open(file_path).convert("RGB")
        image = np.array(image)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label
