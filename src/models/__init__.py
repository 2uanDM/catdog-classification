from torch import nn

from .cnn import CNN


def get_model(model_name: str) -> nn.Module:
    if model_name == "cnn":
        return CNN
    else:
        raise ValueError(f"Model {model_name} not found")


__all__ = ["get_model"]
