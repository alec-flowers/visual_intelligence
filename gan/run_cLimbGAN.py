import numpy as np
import torch
from matplotlib import pyplot as plt

from data.data import get_data
from gan.cGAN import get_device, generate_noise, load_generator
from gan.gan_model import LimbLengthGenerator
from pose.plot import plot_3d_keypoints
from pose.utils import GOOD_POSES_PATH, NOISE_DIMENSION, calc_limb_lengths, PLOT_PATH, CLIMBGAN_PATH


def differences_in_limb_lengths(generated_limb_length, original_limb_length):
    difference = np.linalg.norm(generated_limb_length - original_limb_length)
    return difference


def generate_coords_given_limb_lengths(limb_lengths, label, version, plot=False):
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
        plot_3d_keypoints(coord[:, 0], coord[:, 1], coord[:, 2], elev=-70, azim=270)
    else:
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
    VERSION = 780
    device = get_device()
    train_loader, _, train_coordinate_dataset, _ = get_data(batch_size=64, split_ratio=0.15, path=GOOD_POSES_PATH)

    # Plot generated images conditioned on label and limb length
    for version in range(364, VERSION + 1, 52):
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
        differences = differences_in_limb_lengths(generated_limb_length[0], train_coordinate_dataset.limb_lengths[0])
        mean_differences.append((version, differences))

    plot_limb_length_convergence(mean_differences)
