import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(EfficientNetB0, self).__init__()

        # Load pretrained EfficientNet-B0
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b0(weights=weights)

        # Modify the classifier head for binary classification
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # Instantiate the model
    model = EfficientNetB0()

    # Random input tensor
    x = torch.randn(64, 3, 150, 150)  # (batch_size, channels, height, width)

    # Forward pass
    output = model(x)

    print(output.shape)
