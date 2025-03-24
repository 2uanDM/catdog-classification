import argparse

from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train a model on the given dataset")
    parser.add_argument("--model", type=str, default="cnn", help="Model to train")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory containing the data"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory"
    )
    parser.add_argument(
        "--split_ratio", type=float, default=0.8, help="Train/Val split ratio"
    )
    args = parser.parse_args()

    trainer = Trainer(
        model=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split_ratio=args.split_ratio,
    )


if __name__ == "__main__":
    main()
