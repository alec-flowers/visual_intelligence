import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from gan.cGAN import get_device, generate_noise, load_generator
from gan.gan_models import Generator
from pose.plot import plot_3d_keypoints
from pose.pose_utils import CGAN_PATH


def generate_images(generator: nn.Module, labels: list, device: torch.device) \
        -> torch.Tensor:
    noise_vector = generate_noise(len(labels), 100, device=device)
    generator.eval()
    images = generator((noise_vector, labels))
    return images


def plot_generated_images(generator: nn.Module, labels: list, device: torch.device = get_device()):
    """ Generate subplots with generated examples. """
    images = generate_images(generator, labels, device)
    plt.figure(figsize=(10, 10))
    for i in range(len(labels)):
        image = images[i]
        image = image.cpu().detach().numpy()
        image = np.reshape(image, (33, 3))
        plot_3d_keypoints(image[:, 0], image[:, 1], image[:, 2])


if __name__ == "__main__":
    learning_rate = 0.0002
    version = 728
    labels = torch.tensor([0, 0, 1, 1, 2, 2])  # 0, 1, 2

    device = get_device()
    generator = Generator().to(device)
    generator = load_generator(generator, version, CGAN_PATH)

    images = generate_images(generator, labels, device)

    plot_generated_images(generator, labels=labels, device=get_device())




