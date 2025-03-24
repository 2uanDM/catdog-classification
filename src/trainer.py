import os

import torch
from rich.console import Console
from torch.utils.data import DataLoader

from .models import get_model
from .utils.dataset import CatDogDataset

console = Console()


class Trainer:
    def __init__(
        self,
        model: str = "cnn",
        data_dir: str = "data",
        output_dir: str = "outputs",
        split_ratio: float = 0.8,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(model)().to(self.device)

        console.log(f"Using device: {self.device} to init model: {model.upper()}")

        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Init f

    def init_for_train(self):
        data = CatDogDataset(
            split_ratio=self.split_ratio,
            data_dir=self.data_dir,
        )

        self.train_dataset = data.from_mode(mode="train")
        self.val_dataset = data.from_mode(mode="val")

        console.log(f"Train dataset size: {len(self.train_dataset)}")
        console.log(f"Validation dataset size: {len(self.val_dataset)}")

        # Dataloader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def train(self):
        pass
