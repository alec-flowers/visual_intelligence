import matplotlib.pyplot as plt
import numpy as np


def plot_data(dataloader, n_images=4):
    images, labels = next(iter(dataloader))
    for i in range(n_images):
        image = images[i].numpy().transpose((1, 2, 0))
        plt.subplot(int(np.sqrt(n_images)), int(np.sqrt(n_images)), i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image)
        plt.xlabel(dataloader.dataset.classes[labels[i].item()])
    plt.tight_layout()
    plt.show()


def plot_image_grid(images, n_images):
    for i in range(n_images):
        plt.subplot(int(np.sqrt(n_images)), int(np.sqrt(n_images)), i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
    plt.tight_layout()
    plt.show()

