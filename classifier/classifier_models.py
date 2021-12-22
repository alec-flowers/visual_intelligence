import torch.nn as nn


class MLP(nn.Module):
    """
        Multilayer Perceptron for classifying poses based on detected keypoints.
        Inputs are 33 landmarks with (x,y,z) coordinates, respectively, leading to 33*3 input nodes.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(33 * 3, 64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


class GoodBadMLP(nn.Module):
    """
        Multilayer Perceptron for classifying good-bad poses based on detected keypoints.
        Inputs are 33 landmarks with (x,y,z) coordinates, respectively, leading to 33*3 input nodes.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(33 * 3, 64),
            nn.BatchNorm1d(num_features=64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(num_features=64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(num_features=32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(num_features=16),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        output = self.layers(x)
        return output.view(-1)


class CNN(nn.Module):
    """
    CNN for classifying poses based on the raw input image.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3)),
            nn.SELU(),
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.SELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=2),
            nn.SELU(),
            nn.Flatten(),
            nn.Linear(5721664, num_classes)
        )

    def forward(self, x):
        return self.layers(x)
