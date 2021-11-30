import torch
import torch.nn as nn

N_CLASSES = 3
LATENT_DIM = 100
GENERATOR_OUTPUT_IMAGE_SHAPE = 33 * 3
EMBEDDING_DIM = 100
NOISE_DIMENSION = 100


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


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_conditioned_generator = nn.Sequential(nn.Embedding(N_CLASSES, EMBEDDING_DIM),
                                                         nn.Linear(EMBEDDING_DIM, GENERATOR_OUTPUT_IMAGE_SHAPE))

        self.latent = nn.Sequential(nn.Linear(LATENT_DIM, GENERATOR_OUTPUT_IMAGE_SHAPE),
                                    nn.LeakyReLU(0.2, inplace=True))

        self.model = nn.Sequential(
            # First upsampling
            nn.Linear(GENERATOR_OUTPUT_IMAGE_SHAPE * 2, 128, bias=False),
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.25),
            # Second upsampling
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.25),
            # Third upsampling
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.25),
            # Final upsampling
            nn.Linear(512, GENERATOR_OUTPUT_IMAGE_SHAPE, bias=False),
            nn.Tanh()
        )

    def forward(self, inputs):
        noise_vector, label = inputs

        label_output = self.label_conditioned_generator(label)
        label_output = label_output.view(-1, 33, 3)  # view(-1, 1, 4, 4)

        latent_output = self.latent(noise_vector)
        latent_output = latent_output.view(-1, 33, 3)

        concat = torch.cat((latent_output, label_output), dim=2)
        concat = concat.view(concat.size(0), -1).float()
        image = self.model(concat)
        return image.view(-1, 33, 3)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_condition_disc = nn.Sequential(nn.Embedding(N_CLASSES, EMBEDDING_DIM),
                                                  nn.Linear(EMBEDDING_DIM, GENERATOR_OUTPUT_IMAGE_SHAPE))

        self.model = nn.Sequential(
            nn.Linear(GENERATOR_OUTPUT_IMAGE_SHAPE * 2, 1024),  # Account for label by * 2
            nn.LeakyReLU(0.25),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.25),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.25),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Outputs single value whether image was real or fake
        )

    def forward(self, inputs):
        img, label = inputs
        label_output = self.label_condition_disc(label)
        label_output = label_output.view(-1, 33, 3)
        concat = torch.cat((img, label_output), dim=2)
        concat = concat.view(concat.size(0), -1).float()
        output = self.model(concat)
        return output
