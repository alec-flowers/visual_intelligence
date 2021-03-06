import glob

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from pose.pose_utils import PLOT_PATH


def import_images():
    image_list = []
    for filename in glob.glob(str(PLOT_PATH) + '/*.png'):  # assuming gif
        im = Image.open(filename)
        image_list.append(im)
    return image_list


if __name__ == "__main__":
    image_list = import_images()
    filenames = [img.filename.split('plots/')[1].split('.png')[0] for img in image_list]
    s = np.argsort(filenames)

    # Sort filenames and images
    image_list = [np.array(img) for img in image_list]
    image_list = np.array(image_list)[s]
    filenames = np.array(filenames)[s].tolist()

    fig = plt.figure(figsize=(8, 8))
    # plt.title("Generated images by cLimbGAN")
    plt.axis('off')
    columns = 4
    rows = 4
    OFFSET = 0
    ax = []
    for i in range(OFFSET, OFFSET + columns * rows):
        img = image_list[i]
        t = filenames[i].split('_version_')[1]
        i = i - OFFSET
        ax.append(fig.add_subplot(rows, columns, i+1))
        ax[-1].set_title("After epoch  " + str(int(t)))  # set title
        plt.axis('off')
        plt.imshow(img)
    plt.tight_layout()
    plt.savefig(str(PLOT_PATH) + f"/GAN_convergence_over_epochs.png")
    plt.show()
