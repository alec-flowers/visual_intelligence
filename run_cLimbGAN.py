import numpy as np
import torch
from matplotlib import pyplot as plt

from cGAN import get_device, generate_noise, load_generator
from model import LimbLengthGenerator
from utils import BODY_POSE_CONNECTIONS, GOOD_POSES_PATH, NOISE_DIMENSION
from data import get_data


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
        # plt.axis('off')
        #plt.savefig(f'./plots/test_{i}.jpg')


def plot_coordinates(original_coords, generated_coords, labels):
    """ Generate subplots with generated examples. """
    plt.figure(figsize=(10, 10))
    for i in range(len(labels)):
        # Get image
        original_coord = original_coords[i]
        generated_coord = generated_coords[i]
        # Plot
        #plt.subplot(3, 3, i+1)
        plot_3d_keypoints(original_coord[:, 0], original_coord[:, 1], original_coord[:, 2])
        plot_3d_keypoints(generated_coord[:, 0], generated_coord[:, 1], generated_coord[:, 2])


if __name__ == "__main__":
    learning_rate = 0.0002
    version = 468
    N = 6
    device = get_device()
    generator = LimbLengthGenerator().to(device)
    generator = load_generator(generator, version, "./saved_model/cLimbGAN")
    train_loader, _, train_coordinate_dataset, _ = get_data(batch_size=64, split_ratio=0.15, path=GOOD_POSES_PATH)

    noise_vector = generate_noise(N, NOISE_DIMENSION, device=device)
    labels = torch.tensor(train_coordinate_dataset.labels[:N]).unsqueeze(1).long()  # 0, 1, 2

    generated_coords = generator((noise_vector, labels, torch.tensor(train_coordinate_dataset.limb_lengths[:N])))

    plot_coordinates(train_coordinate_dataset.coordinates[:N], generated_coords.detach().numpy(), labels)
    plot_coordinates(train_coordinate_dataset.coordinates[:N], generated_coords.detach().numpy(), labels)

    # plot_generated_images(generator, labels=labels, device=get_device())




