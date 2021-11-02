import matplotlib.pyplot as plt
import numpy as np
import math


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


def plot_image(image, dataloader=False, label=None, title=''):
    if dataloader:
        image = image.numpy().transpose((1, 2, 0))
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image)
    plt.xlabel(label)
    plt.tight_layout()
    plt.title(title)
    plt.show()


def plot_image_grid(images, n_images, dataloader=False, title=""):
    for i in range(n_images):
        if dataloader:
            image = images[i].numpy().transpose((1, 2, 0))
        else:
            image = images[i]
        plt.subplot(math.ceil(np.sqrt(n_images)), math.ceil(np.sqrt(n_images)), i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.suptitle(title, size=16)
    plt.show()


def plot_dataset_images(dataset, num):
    images, labels = [], []
    i = 0
    for image, label in dataset:
        i += 1
        if i > len(dataset):
            break
        images.append(image)
        labels.append(labels)
        if i >= num: break

    plot_image_grid(images, len(images), dataloader=True, title="Photos from raw dataset")


def plot_annotated_images(annotated_images, num):
    annotated_plot = []
    i = 0
    j = 0
    while i < num:
        if i >= len(annotated_images):
            break
        if annotated_images[j] is None:
            j = j + 1
            continue
            #annotated_plot.append([[0, 0], [0, 0]])
        else:
            annotated_plot.append(annotated_images[j])
            j = j + 1
            i = i + 1

    plot_image_grid(annotated_plot[:num], len(annotated_plot[:num]), title="Annotated poses")


def plot_no_pose_photo(df, dataset, num=9):
    indx_null = df.index[df["NOSE"].isnull()].tolist()

    bad_photos = []
    i = 0
    for ind in indx_null:
        if i >= num: break
        bad_photos.append(dataset[ind][0])
        i += 1

    plot_image_grid(bad_photos, len(bad_photos), dataloader=True, title="Photos where no pose was detected")
