import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt

from cGAN import get_device, generate_noise
from model import Generator, Discriminator
from utils import BODY_POSE_CONNECTIONS


def plot_3d_keypoints(x, y, z):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.view_init(-70, -95)
    ax.scatter3D(x, y, z)
    for i, j in BODY_POSE_CONNECTIONS:
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color='b')
    plt.show()


def plot_generated_images(generator, pose, device=get_device()):
    """ Generate subplots with generated examples. """
    images = []
    labels = torch.tensor([0, 0, 0]) + pose
    noise_vector = generate_noise(len(labels), 100, device=device)

    generator.eval()
    images = generator((noise_vector, labels))
    plt.figure(figsize=(10, 10))
    for i in range(len(labels)):
        # Get image
        image = images[i]
        # Convert image back onto CPU and reshape
        image = image.cpu().detach().numpy()
        image = np.reshape(image, (33, 3))
        # Plot
        # plt.subplot(2, 2, i+1)
        plot_3d_keypoints(image[:, 0], image[:, 1], image[:, 2])
        # plt.axis('off')
        #plt.savefig(f'./plots/test_{i}.jpg')


if __name__ == "__main__":
    learning_rate = 0.0002
    version = 572
    pose = 0  # 0, 1, 2

    device = get_device()
    generator = Generator().to(device)
    generator.load_state_dict(torch.load("./saved_model/cGAN" + f'/generator_{version}.pth'))

    plot_generated_images(generator, pose=pose, device=get_device())





