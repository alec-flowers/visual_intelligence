import numpy as np
import torch
from matplotlib import pyplot as plt

from data.data_loading import get_data
from gan.cGAN import get_device, generate_noise, load_generator
from gan.gan_models import LimbLengthGenerator
from pose.plot import plot_3d_keypoints
from pose.pose_utils import GOOD_POSES_PATH, NOISE_DIMENSION, calc_limb_lengths, PLOT_PATH, CLIMBGAN_PATH


def differences_in_limb_lengths(generated_limb_length: np.array, original_limb_length: np.array) -> float:
    """
    Calculate the differences in limb lengths from the generated to the original image to check
    if GAN converges.
    :param generated_limb_length:
    :type generated_limb_length:
    :param original_limb_length:
    :type original_limb_length:
    :return: the differences
    :rtype: float
    """
    difference = np.linalg.norm(generated_limb_length - original_limb_length)
    return difference


def generate_coords_given_limb_lengths(limb_lengths: np.array,
                                       label: int,
                                       version: int,
                                       plot: bool = False) -> np.array:
    """
    Generate the coordinates of a pose given the label and the limb lengths of a person.
    :param limb_lengths:  the person's limb lengths
    :type limb_lengths: np.array
    :param label: the desired pose label
    :type label: int
    :param version: the version of the GAN to consult
    :type version: int
    :param plot: whether to plot the generated image
    :type plot: bool
    :return: the generated limb lengths
    :rtype: np.array
    """
    device = get_device()
    generator = LimbLengthGenerator().to(device)
    generator = load_generator(generator, version, CLIMBGAN_PATH)
    generator.eval()
    noise_vector = generate_noise(number_of_images=1, noise_dimension=NOISE_DIMENSION, device=device)
    labels = torch.tensor([label], dtype=torch.long).unsqueeze(1)  # 0, 1, 2
    limb_lengths = torch.tensor(np.array(limb_lengths)).unsqueeze_(0)
    generated_coords = generator((noise_vector, labels, limb_lengths))
    generated_limb_lengths = [calc_limb_lengths(coords) for coords in generated_coords.detach().numpy()]

    if plot:
        # Plot generated image given the original limb lengths
        coord = generated_coords.detach().numpy().squeeze(0)
        plot_3d_keypoints(coord[:, 0], coord[:, 1], coord[:, 2], elev=-70, azim=270, version=version)
    else:
        return generated_limb_lengths[0]


def plot_limb_length_convergence(mean_differences: list):
    plt.plot(*zip(*mean_differences))
    # plt.title("Mean squared average deviation of generated from target limb lengths")
    plt.xlabel("Training epoch")
    plt.ylabel("Mean squared average deviation")
    plt.ylim(0, 2)
    plt.savefig(str(PLOT_PATH) + f"/mean_differences.png")
    plt.show()


if __name__ == "__main__":
    VERSION = 1092
    train_loader, _, train_coordinate_dataset, _ = get_data(batch_size=64, split_ratio=0.15, path=GOOD_POSES_PATH)

    # Plot generated images conditioned on label and limb length
    for version in range(104, VERSION + 1, 52):
        generate_coords_given_limb_lengths(calc_limb_lengths(train_coordinate_dataset.coordinates[0]),
                                           train_coordinate_dataset.labels[0],
                                           version=version,
                                           plot=True)

    # Check if the mean differences from generated to input image decrease over time
    mean_differences = []
    for version in range(104, VERSION + 1, 52):
        generated_limb_length = \
            generate_coords_given_limb_lengths(calc_limb_lengths(train_coordinate_dataset.coordinates[0]),
                                               train_coordinate_dataset.labels[0],
                                               version=version,
                                               plot=False)
        differences = differences_in_limb_lengths(generated_limb_length, train_coordinate_dataset.limb_lengths[0])
        mean_differences.append((version, differences))

    plot_limb_length_convergence(mean_differences)
