import numpy as np
import torch
from matplotlib import pyplot as plt

from gan.cGAN import get_device, generate_noise, load_generator
from data.data import get_data
from gan.gan_model import LimbLengthGenerator
from utils import BODY_POSE_CONNECTIONS, GOOD_POSES_PATH, NOISE_DIMENSION, calc_limb_lengths, PLOT_PATH


def plot_3d_keypoints(x, y, z, title=""):
    # fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.view_init(-70, -95)
    ax.scatter3D(x, y, z)
    for i, j in BODY_POSE_CONNECTIONS:
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color='b')
    plt.title(title)
    plt.show()


def plot_coordinates(coords, labels, version=None):
    """ Generate subplots with generated examples. """
    fig = plt.figure()
    for i in range(len(labels)):
        # Get image
        coord = coords[i]
        # Plot
        if version:
            plot_3d_keypoints(coord[:, 0], coord[:, 1], coord[:, 2],
                              title=f"Image {i} with label {labels[i].item()}, generator version {version}")
            fig.figure.savefig(str(PLOT_PATH) + f"/img_{i}_generator_vs_{version}.png")
        else:
            plot_3d_keypoints(coord[:, 0], coord[:, 1], coord[:, 2],
                              title=f"Original pose estimate {i} with label {labels[i].item()}")
            fig.figure.savefig(str(PLOT_PATH) + f"/original_img_{i}_label_{labels[i].item()}.png")


def differences_in_limb_lengths(generated_limb_lengths ,train_coordinate_dataset, start, stop):
    # Compare limb lengths
    differences = np.mean([np.linalg.norm(generated_limb_length - original_limb_length)
                           for (generated_limb_length, original_limb_length)
                           in zip(generated_limb_lengths, train_coordinate_dataset.limb_lengths[start:stop])])

    return differences


def generate_coords_given_limb_lengths(train_coordinate_dataset, start, stop, version, plot=False):
    generator = LimbLengthGenerator().to(device)
    generator = load_generator(generator, version, "./saved_model/cLimbGAN")

    noise_vector = generate_noise(stop-start, NOISE_DIMENSION, device=device)
    labels = torch.tensor(train_coordinate_dataset.labels[start:stop]).unsqueeze(1).long()  # 0, 1, 2

    generated_coords = generator((noise_vector, labels,
                                  torch.tensor(np.array(train_coordinate_dataset.limb_lengths[start:stop]))))
    generated_limb_lengths = [calc_limb_lengths(coords) for coords in generated_coords.detach().numpy()]

    if plot:
        # Plot generated image given the original limb lengths
        plot_coordinates(generated_coords.detach().numpy(), labels, version)

    return generated_limb_lengths


def plot_limb_length_convergence(mean_differences):
    plt.plot(*zip(*mean_differences))
    plt.title("Mean squared average deviation of generated from target limb lengths")
    plt.xlabel("Training epoch")
    plt.ylabel("Mean squared average deviation")
    plt.ylim(0, 2)
    plt.savefig(str(PLOT_PATH) + f"/mean_differences.png")
    plt.show()


if __name__ == "__main__":
    learning_rate = 0.0002
    VERSION = 780
    START = 22
    STOP = 24
    device = get_device()

    train_loader, _, train_coordinate_dataset, _ = get_data(batch_size=64, split_ratio=0.15, path=GOOD_POSES_PATH)
    # d = generate_coords_given_limb_lengths(train_coordinate_dataset, START, STOP, VERSION, plot=False)

    # Plot original image
    plot_coordinates(train_coordinate_dataset.coordinates[START:STOP], train_coordinate_dataset.labels[START:STOP], None)

    # Plot generated images conditioned on label and limb length
    for version in range(52, VERSION+1, 52):
        generate_coords_given_limb_lengths(train_coordinate_dataset, START, STOP, version, plot=True)

    # Check if the mean differences from generated to input image decrease over time
    mean_differences = []
    for version in range(52, VERSION+1, 52):
        generated_limb_lengths = generate_coords_given_limb_lengths(train_coordinate_dataset, START, STOP, version, plot=False)
        differences = differences_in_limb_lengths(generated_limb_lengths, train_coordinate_dataset, START, STOP)
        mean_differences.append((version, differences))

    plot_limb_length_convergence(mean_differences)



