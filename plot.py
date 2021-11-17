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


def plot_image_grid(images, n_images, dataloader=False, title="", subplot_title=[]):
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
    #plt.savefig('annotated_images.jpg')
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


def find_misclassified_images(difference, all_images, type_, start_index):
    indices = np.where(difference != 0)
    np.random.shuffle(indices[0])
    misclassified_images = None
    if type_ == 'validation':
        misclassified_images = [all_images[i+start_index] for i in indices[0]]
    elif type_ == 'training':
        misclassified_images = [all_images[i] for i in indices[0]]
    return misclassified_images, indices


def find_correctly_classified_images(difference, all_images, type_, start_index):
    indices = np.where(difference == 0)
    np.random.shuffle(indices[0])
    classified_images = None
    if type_ == 'validation':
        classified_images = [all_images[i+start_index] for i in indices[0]]
    elif type_ == 'training':
        classified_images = [all_images[i] for i in indices[0]]
    return classified_images, indices


def plot_classfied_images(targets, predictions, all_images, type_="training",
                          max_n_to_plot=16, classified="misclassified", split_ratio=0.8):
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

