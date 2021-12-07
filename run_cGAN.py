import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt

from cGAN import get_device, generate_noise, load_generator
from model import Generator, Discriminator
from plot import plot_image_grid
from utils import BODY_POSE_CONNECTIONS


def plot_3d_keypoints(x, y, z):
    # fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.view_init(-70, -95)
    ax.scatter3D(x, y, z)
    for i, j in BODY_POSE_CONNECTIONS:
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color='b')
    plt.show()


def generate_images(generator, labels, device):
    noise_vector = generate_noise(len(labels), 100, device=device)
    generator.eval()
    images = generator((noise_vector, labels))

    return images


def plot_generated_images(generator, labels, device=get_device()):
    """ Generate subplots with generated examples. """
    images = generate_images(generator, labels, device)
    plt.figure(figsize=(10, 10))
    for i in range(len(labels)):
        # Get image
        image = images[i]
        # Convert image back onto CPU and reshape
        image = image.cpu().detach().numpy()
        image = np.reshape(image, (33, 3))
        # Plot
        #plt.subplot(3, 3, i+1)
        plot_3d_keypoints(image[:, 0], image[:, 1], image[:, 2])



if __name__ == "__main__":
    learning_rate = 0.0002
    version = 364
    labels = torch.tensor([0, 0, 1, 1, 2, 2])  # 0, 1, 2

    device = get_device()
    generator = Generator().to(device)
    generator = load_generator(generator, version, "./saved_model/cGAN")

    images = generate_images(generator, labels, device)

    plot_generated_images(generator, labels=labels, device=get_device())

    # plot_generated_images(generator, labels=labels, device=get_device())




