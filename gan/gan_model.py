import torch
import torch.nn as nn
from pose.utils import BODY_POSE_CONNECTIONS

N_CLASSES = 3
LATENT_DIM = 50
GENERATOR_OUTPUT_IMAGE_SHAPE = 33 * 3
EMBEDDING_DIM = 100

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
        coordinates, label = inputs
        label_output = self.label_condition_disc(label)
        label_output = label_output.view(-1, 33, 3)
        concat = torch.cat((coordinates, label_output), dim=2)
        concat = concat.view(concat.size(0), -1).float()
        output = self.model(concat)
        return output


class LimbLengthGenerator(nn.Module):
    def __init__(self):
        super(LimbLengthGenerator, self).__init__()

        self.label_condition_gen = nn.Sequential(nn.Embedding(N_CLASSES, 10),
                                                 nn.Linear(10, GENERATOR_OUTPUT_IMAGE_SHAPE))

        self.limb_lengths_gen = nn.Sequential(nn.Linear(len(BODY_POSE_CONNECTIONS), GENERATOR_OUTPUT_IMAGE_SHAPE))

        self.latent = nn.Sequential(nn.Linear(GENERATOR_OUTPUT_IMAGE_SHAPE, GENERATOR_OUTPUT_IMAGE_SHAPE),
                                    nn.LeakyReLU(0.2, inplace=True))

        self.model = nn.Sequential(
            # First upsampling
            nn.Linear(GENERATOR_OUTPUT_IMAGE_SHAPE * 3, 128, bias=False),
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
        noise_vector, label, limb_lengths = inputs
        label_output = self.label_condition_gen(label)
        label_output = label_output.view(-1, 33, 3)

        limb_lengths_output = self.limb_lengths_gen(limb_lengths.float())
        limb_lengths_output = limb_lengths_output.view(-1, 33, 3)

        latent_output = self.latent(noise_vector)
        latent_output = latent_output.view(-1, 33, 3)

        combined_input = torch.cat((label_output, limb_lengths_output, latent_output), dim=2).float()
        combined_input = combined_input.view(combined_input.size(0), -1).float()
        generated_coordinates = self.model(combined_input)
        return generated_coordinates.view(-1, 33, 3)


class LimbLengthDiscriminator(nn.Module):
    def __init__(self):
        super(LimbLengthDiscriminator, self).__init__()

        self.label_condition_disc = nn.Sequential(nn.Embedding(N_CLASSES, 10),
                                                  nn.Linear(10, GENERATOR_OUTPUT_IMAGE_SHAPE))

        self.limb_lengths_disc = nn.Sequential(nn.Linear(len(BODY_POSE_CONNECTIONS), GENERATOR_OUTPUT_IMAGE_SHAPE))

        self.model = nn.Sequential(
            nn.Linear(GENERATOR_OUTPUT_IMAGE_SHAPE * 3, 1024),  # Account for label by * 2
            nn.LeakyReLU(0.25),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.25),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.25),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Outputs single value whether image was real or fake
        )

    def forward(self, inputs):
        coordinates, label, limb_lengths = inputs

        label_output = self.label_condition_disc(label)
        label_output = label_output.view(-1, 33, 3)

        limb_lengths_output = self.limb_lengths_disc(limb_lengths.float())
        limb_lengths_output = limb_lengths_output.view(-1, 33, 3)

        combined_input = torch.cat((label_output, limb_lengths_output, coordinates), dim=2)
        combined_input = combined_input.view(combined_input.size(0), -1).float()

        real_or_fake = self.model(combined_input)
        return real_or_fake
