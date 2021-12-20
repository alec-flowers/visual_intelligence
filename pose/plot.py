import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from data.data_loading import ClassifyDataset
from pose.pose_utils import PLOT_PATH, BODY_POSE_CONNECTIONS


def plot_image(image: np.array, dataloader: bool = False, label: bool = None, title: str = ''):
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


def plot_image_grid(images: np.array, n_images: int, dataloader: bool = False, title: str = "",
                    subplot_title: list =[]):
    for i in range(n_images):
        if dataloader:
            image = images[i].numpy().transpose((1, 2, 0))
        else:
            image = images[i]
        if subplot_title:
            plt.subplot(math.ceil(np.sqrt(n_images)), math.ceil(np.sqrt(n_images)), i + 1).set_title(f"t: {subplot_title[i][0]}, p: {subplot_title[i][1]}")
        else:
            plt.subplot(math.ceil(np.sqrt(n_images)), math.ceil(np.sqrt(n_images)), i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.suptitle(title, size=16)
    plt.show()


def plot_dataset_images(dataset: ClassifyDataset, num: int):
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


def plot_annotated_images(annotated_images: np.array, num: int):
    annotated_plot = []
    i = 0
    j = 0
    while i < num:
        if i >= len(annotated_images):
            break
        if annotated_images[j] is None:
            j = j + 1
            continue
        else:
            annotated_plot.append(annotated_images[j])
            j = j + 1
            i = i + 1

    plot_image_grid(annotated_plot[:num], len(annotated_plot[:num]), title="Annotated poses")


def plot_no_pose_photo(df: pd.DataFrame, dataset: ClassifyDataset, num:int = 9):
    indx_null = df.index[df["NOSE"].isnull()].tolist()

    bad_photos = []
    i = 0
    for ind in indx_null:
        if i >= num: break
        bad_photos.append(dataset[ind][0])
        i += 1

    plot_image_grid(bad_photos, len(bad_photos), dataloader=True, title="Photos where no pose was detected")


def find_misclassified_images(difference: torch.Tensor, all_images: np.array, type_: str, start_index: int) \
        -> Tuple[np.array, np.array]:
    indices = np.where(difference != 0)
    np.random.shuffle(indices[0])
    misclassified_images = None
    if type_ == 'validation':
        misclassified_images = [all_images[i+start_index] for i in indices[0]]
    elif type_ == 'training':
        misclassified_images = [all_images[i] for i in indices[0]]
    return misclassified_images, indices


def find_correctly_classified_images(difference: torch.Tensor, all_images: np.array, type_: str, start_index: int) \
        -> Tuple[np.array, np.array]:
    indices = np.where(difference == 0)
    np.random.shuffle(indices[0])
    classified_images = None
    if type_ == 'validation':
        classified_images = [all_images[i+start_index] for i in indices[0]]
    elif type_ == 'training':
        classified_images = [all_images[i] for i in indices[0]]
    return classified_images, indices


def plot_classified_images(targets: torch.Tensor,
                           predictions: torch.Tensor,
                           all_images: np.array,
                           type_: str = "training",
                           max_n_to_plot: int = 16,
                           classified: str = "misclassified",
                           split_ratio: float = 0.8):
    indices = None
    start_index = None
    classified_images = None
    if type_ == 'validation':
        start_index = int(split_ratio * len(all_images))
    # Extract images that are miss-predicted
    difference = targets - predictions
    if classified == 'misclassified':
        classified_images, indices = find_misclassified_images(difference, all_images, type_, start_index)
    elif classified == 'correctly classified':
        classified_images, indices = find_correctly_classified_images(difference, all_images, type_, start_index)
    print(f"{len(classified_images)} out of {len(difference)} {type_} images were {classified}.")

    # Create subplot titles, consisting of targets and predictions
    subplot_title = list(zip([targets[i].item() for i in indices[0]],
                             [predictions[i].item() for i in indices[0]]))
    plot_n_images = min(len(classified_images), max_n_to_plot)
    plot_image_grid(classified_images[:plot_n_images], plot_n_images, dataloader=False,
                    title=f"{classified} {type_} images", subplot_title=subplot_title)


def plot_confusion_matrix(targets: torch.Tensor, predicted: torch.Tensor, title: str = None, save_plot: bool = False):
    class_names = ["DD", "W1", "W2"]
    conf_mat = confusion_matrix(targets.numpy(), predicted.numpy())
    fig = sns.heatmap(conf_mat, annot=True,
                      cmap=sns.color_palette("light:#5A9", as_cmap=True),
                      cbar_kws={'label': 'count'}, fmt='g')
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix for the " + title + " data")
    tick_marks = np.arange(len(class_names)) + 0.5
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names, rotation=0)
    plt.tight_layout()
    if save_plot:
        fig.figure.savefig(PLOT_PATH + title + "_data_confusion_matrix.png")
    plt.show()


def plot_3d_keypoints(x: np.array, y: np.array, z: np.array, elev: int = -50, azim: int = 270, version: int = 1):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(x, y, z)
    ax.view_init(elev=elev, azim=azim)
    for i,j in BODY_POSE_CONNECTIONS:
        ax.plot([x[i],x[j]], [y[i],y[j]], [z[i],z[j]], color='b')
    plt.savefig(str(PLOT_PATH) + f"/generated_version_0{version}.png")
    plt.show()


def plot_distribution(df: pd.DataFrame, angles: np.array, landmarks: list):
    fig, ax = plt.subplots(2, 8, figsize=(30, 15))
    sns.set_palette(sns.color_palette("Set1"))
    for idx, col in enumerate(landmarks):
        y = int(idx/8)
        x = idx % 8
        sns.histplot(df, x=col, hue='quality', ax=ax[y, x], bins=10, multiple='dodge')
        ax[y, x].vlines(angles[col][0], 0, 200)
        ax[y, x].vlines(angles[col][2], 0, 200)
        ax[y, x].tick_params(axis='x')
        ax[y, x].set_ylim(0,200)
        ax[y, x].set_xlim(0,180)


def plot_distribution_with_image(df: pd.DataFrame, df_new: pd.DataFrame, angles: np.array, landmarks):
    fig, ax = plt.subplots(2, 8, figsize=(30, 15))
    sns.set_palette(sns.color_palette("Set1"))
    for idx, col in enumerate(landmarks):
        y = int(idx/8)
        x = idx % 8
        sns.histplot(df, x=col, hue='quality', ax=ax[y, x], bins=10, multiple='dodge')
        ax[y, x].vlines(angles[col][0], 0, 200)
        ax[y, x].vlines(df_new[col][0], 0, 200, colors='b', linestyles='dashed')
        ax[y, x].vlines(angles[col][2], 0, 200)
        ax[y, x].tick_params(axis='x')
        ax[y, x].set_ylim(0, 200)
        ax[y, x].set_xlim(0, 180)
    plt.show()

    for ang in angles:
        low = angles[ang][0]
        high = angles[ang][2]
        current = df_new[ang][0]
        if current < low:
            print(f"{ang} is too small by - {low - current:.1f} degrees")
        elif current > high:
            print(f"{ang} is too large by - {current - high:.1f} degrees")