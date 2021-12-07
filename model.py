from torch import nn


class MLP(nn.Module):
    """
        Multilayer Perceptron.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(33 * 3, 64),  # Input are 33 landmarks with (x,y,z) coordinates, respectively
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)


class CNN(nn.Module):
    """
    CNN
    """
    def __init__(self, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            # transforms.ConvertImageDtype(torch.float32),
            # transforms.Grayscale(num_output_channels=3),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.SELU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.SELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.SELU(),
            nn.Flatten(),
            nn.Linear(5721664, num_classes)
        )

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)
