import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layers with BatchNorm
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (64, 75, 75)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (128, 37, 37)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # (256, 18, 18)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        self.fc2 = nn.Linear(512, 2)  # Binary classification

    def forward(self, x):
        # Conv + BatchNorm + ReLU6
        x = F.relu6(self.bn1(self.conv1(x)))
        x = F.relu6(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu6(self.bn3(self.conv3(x)))
        x = F.relu6(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        x = F.relu6(self.bn5(self.conv5(x)))
        x = F.relu6(self.bn6(self.conv6(x)))
        x = self.pool3(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu6(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)

        return x  # No softmax, use nn.CrossEntropyLoss()


if __name__ == "__main__":
    # Instantiate the model
    model = CNN()

    # Random input tensor
    x = torch.randn(64, 3, 150, 150)  # (batch_size, channels, height, width)

    # Forward pass
    output = model(x)

    print(output.shape)
