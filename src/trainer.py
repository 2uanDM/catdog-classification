import time
from pathlib import Path

import torch
import wandb
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

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
        epochs: int = 10,
        lr: float = 1e-3,
        log_interval: int = 10,
        wandb_key: str = "",
        save_best: bool = True,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(model)().to(self.device)
        self.model_name = model.upper()

        console.log(f"Using device: {self.device} to init model: {self.model_name}")

        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.lr = lr
        self.log_interval = log_interval
        self.wandb_key = wandb_key
        self.save_best = save_best
        self.best_val_accuracy = 0.0

    def init_wandb(self):
        """Initialize Weights & Biases logging with enhanced configuration"""
        run_name = f"{self.model_name}-bs{self.batch_size}-lr{self.lr}-e{self.epochs}"

        config = {
            "model": self.model_name,
            "batch_size": self.batch_size,
            "learning_rate": self.lr,
            "epochs": self.epochs,
            "device": str(self.device),
            "dataset_dir": self.data_dir,
            "split_ratio": self.split_ratio,
        }

        wandb.login(key=self.wandb_key)

        wandb.init(
            project="cat-dog-classification",
            name=run_name,
            config=config,
            dir=self.output_dir,
        )

        wandb.watch(
            self.model,
            log="all",
            log_graph=True,
            criterion=self.criterion,
        )

        # Log model architecture as string
        wandb.run.summary["model_architecture"] = str(self.model)

    def init_before_train(self):
        """Initialize datasets, dataloaders, and optimization components"""
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
            pin_memory=True if self.device.type == "cuda" else False,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        # Criterion and optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2, verbose=True
        )

    def save_checkpoint(self, epoch, val_accuracy, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_accuracy": val_accuracy,
        }

        # Regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Best model checkpoint if applicable
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            console.log(
                f"[bold green]New best model saved![/bold green] Accuracy: {val_accuracy:.2f}%"
            )

        # Log checkpoint to wandb
        if is_best:
            wandb.save(str(best_path))

    def log_images(self, images, labels, outputs, mode="train", epoch=0):
        """Log sample images with predictions to wandb"""
        if epoch % 5 != 0:  # Log images every 5 epochs to avoid clutter
            return

        # Get predictions
        _, preds = torch.max(outputs, 1)

        # Convert labels to readable format
        class_names = ["cat", "dog"]
        true_labels = [class_names[label] for label in labels.cpu().numpy()]
        pred_labels = [class_names[pred] for pred in preds.cpu().numpy()]

        # Create caption for each image: True: X, Pred: Y
        captions = [
            f"True: {true}, Pred: {pred}"
            for true, pred in zip(true_labels, pred_labels)
        ]

        # Create grid of images
        grid = make_grid(images[:16].cpu(), nrow=4, normalize=True)

        # Log to wandb
        wandb.log(
            {
                f"{mode}_images": wandb.Image(
                    grid,
                    caption=f"Epoch {epoch + 1} {mode} images",
                    masks={
                        "predictions": {
                            "mask_data": (preds[:16] == labels[:16]).cpu().numpy(),
                            "class_labels": {0: "incorrect", 1: "correct"},
                        }
                    },
                )
            }
        )

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        batch_time = 0.0
        start_time = time.time()

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            train_task = progress.add_task(
                f"[cyan]Epoch {epoch + 1}/{self.epochs} Training",
                total=len(self.train_loader),
            )

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Calculate metrics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                batch_time += time.time() - start_time
                start_time = time.time()

                # Log batch metrics
                if (batch_idx + 1) % self.log_interval == 0:
                    batch_accuracy = 100.0 * correct / total
                    wandb.log(
                        {
                            "batch/train_loss": loss.item(),
                            "batch/train_accuracy": batch_accuracy,
                            "batch/learning_rate": self.optimizer.param_groups[0]["lr"],
                            "batch/time": batch_time / self.log_interval,
                        }
                    )
                    batch_time = 0.0

                # # Log sample images
                # if batch_idx == 0:
                #     self.log_images(images, labels, outputs, mode="train", epoch=epoch)

                progress.update(train_task, advance=1)

        # Epoch metrics
        train_loss = train_loss / len(self.train_loader)
        train_accuracy = 100.0 * correct / total

        return train_loss, train_accuracy

    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            with Progress(
                TextColumn("[bold green]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ) as progress:
                val_task = progress.add_task(
                    f"[green]Epoch {epoch + 1}/{self.epochs} Validation",
                    total=len(self.val_loader),
                )

                for batch_idx, (images, labels) in enumerate(self.val_loader):
                    images, labels = images.to(self.device), labels.to(self.device)

                    # Forward pass
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    # Calculate metrics
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    # # Log sample images
                    # if batch_idx == 0:
                    #     self.log_images(
                    #         images, labels, outputs, mode="val", epoch=epoch
                    #     )

                    progress.update(val_task, advance=1)

        # Epoch metrics
        val_loss = val_loss / len(self.val_loader)
        val_accuracy = 100.0 * correct / total

        return val_loss, val_accuracy

    def train(self):
        """Run the full training process with enhanced logging and checkpoints"""
        # Initialization
        self.init_before_train()
        self.init_wandb()

        console.log("[bold]Starting training...[/bold]")

        for epoch in range(self.epochs):
            # Train
            train_loss, train_accuracy = self.train_epoch(epoch)

            # Validate
            val_loss, val_accuracy = self.validate(epoch)

            # Update LR scheduler
            self.scheduler.step(val_accuracy)

            # Log epoch metrics
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/accuracy": train_accuracy,
                    "val/loss": val_loss,
                    "val/accuracy": val_accuracy,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }
            )

            # Save checkpoint
            is_best = val_accuracy > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_accuracy

            if self.save_best and is_best:
                self.save_checkpoint(epoch + 1, val_accuracy, is_best=True)
            elif (epoch + 1) % 5 == 0:  # Save regular checkpoint every 5 epochs
                self.save_checkpoint(epoch + 1, val_accuracy, is_best=False)

            # Print epoch summary
            console.log(
                f"[bold]Epoch {epoch + 1}/{self.epochs}[/bold] | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%"
            )

        # Save final model
        final_path = self.output_dir / "final_model.pt"
        torch.save(self.model.state_dict(), final_path)
        wandb.save(str(final_path))

        console.log(
            f"[bold green]Training completed![/bold green] Best accuracy: {self.best_val_accuracy:.2f}%"
        )
        wandb.run.summary["best_accuracy"] = self.best_val_accuracy
        wandb.finish()
