from torch import nn

from .cnn import CNN
from .efficientnet import EfficientNetB0


def get_model(model_name: str) -> nn.Module:
    if model_name == "cnn":
        return CNN
    elif model_name == "efficientnet-b0":
        return EfficientNetB0
    else:
        raise ValueError(f"Model {model_name} not found")


__all__ = ["get_model"]
