import argparse

from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train a model on the given dataset")
    parser.add_argument("--model", "-m", type=str, default="cnn", help="Model to train")
    parser.add_argument(
        "--data_dir",
        "-d",
        type=str,
        default="data",
        help="Directory containing the data",
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", help="Output directory"
    )
    parser.add_argument(
        "--split_ratio", "-s", type=float, default=0.8, help="Train/Val split ratio"
    )
    parser.add_argument(
        "--num_workers",
        "-w",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=0.001,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--num_epochs", "-e", type=int, default=10, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--wandb_key", type=str, default=None, help="Wandb API key for logging"
    )

    args = parser.parse_args()

    trainer = Trainer(
        model=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split_ratio=args.split_ratio,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        num_workers=args.num_workers,
        epochs=args.num_epochs,
        save_best=True,
        wandb_key=args.wandb_key,
    )

    trainer.train()


if __name__ == "__main__":
    main()
