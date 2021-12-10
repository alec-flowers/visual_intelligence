import os
from typing import Any, Callable, Optional, Tuple, List, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils import load_pickle, PICKLEDPATH, TRAINPATH, calc_limb_lengths

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class CoordinatesDataset(Dataset):
    def __init__(self, coordinates, labels, set_type, shuffle=True, split_ratio=0.8):
        if shuffle:
            np.random.seed(42)
            shuffled_indices = np.random.permutation(coordinates.shape[0])
            coordinates = coordinates[shuffled_indices]
            labels = labels[shuffled_indices]
            self.index_order = shuffled_indices
        split_idx = int(split_ratio * coordinates.shape[0])

        if set_type == "train":
            self.coordinates = coordinates[:split_idx]
            self.labels = labels[:split_idx]
            self.limb_lengths = [calc_limb_lengths(coords) for coords in self.coordinates]

        elif set_type == "val":
            self.coordinates = coordinates[split_idx:]
            self.labels = labels[split_idx:]
            self.limb_lengths = [calc_limb_lengths(coords) for coords in self.coordinates]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        assert 0 <= idx < len(self)
        coordinates = self.coordinates[idx]
        label = self.labels[idx]
        limb_lengths = self.limb_lengths[idx]

        return coordinates, label, limb_lengths


class ClassifyDataset(ImageFolder):
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any] = pil_loader,
            transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None
    ):
        super(ImageFolder, self).__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform)

        # drop good/bad/unsure label from classes
        classes, _ = self.find_classes(self.root)
        self.classes = list({cls.split("_")[-1]: True for cls in classes}.keys())

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(
            entry.name for entry in os.scandir(directory) if entry.is_dir() and not entry.name.startswith("2_"))
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        # Map classes to [downwardDog, warrior1, warrior2]
        class_to_idx = {cls_names: i % 3 for i, cls_names in enumerate(classes)}

        return classes, class_to_idx


def load_data(path=TRAINPATH, resize=False, batch_size=32, shuffle=False, batch_sampler=None, subset=False,
              subset_size=100) -> Tuple[ClassifyDataset, DataLoader]:
    if resize:
        resize_size = 300
        transform = (transforms.Compose([transforms.Resize(resize_size),
                                         transforms.CenterCrop(resize_size - 1),
                                         transforms.Grayscale(num_output_channels=3),
                                         transforms.ToTensor(),
                                         transforms.ConvertImageDtype(torch.float32)]))
    else:
        transform = (transforms.Compose([  # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.uint8)]))
    dataset = ClassifyDataset(path, transform=transform)
    if subset:
        indices = torch.arange(subset_size)
        dataset = Subset(dataset, indices)
        shuffle = True

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, batch_sampler=batch_sampler)

    return dataset, dataloader


def train_val_split(images, labels, batch_size=32, shuffle=True, split_ratio=0.8):

    train_dataset = CoordinatesDataset(images, labels, set_type="train", shuffle=shuffle, split_ratio=split_ratio)
    val_dataset = CoordinatesDataset(images, labels, set_type="val", shuffle=shuffle, split_ratio=split_ratio)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=200, shuffle=shuffle)
    return train_dataloader, val_loader, train_dataset, val_dataset


def get_not_none_annotated_images() -> list:
    """
    Not all the images have a corresponding pose estimate.
    In order to plot the data, we need to filter out these images.
    :return: filtered images
    :rtype: list
    """
    annotated_images = load_pickle(PICKLEDPATH, "annotated_images.pickle")
    annotated_images_filtered = []
    for image in annotated_images:
        if image is not None:
            annotated_images_filtered.append(image)
    return annotated_images_filtered


def get_data(batch_size: int, split_ratio: float, path: str):
    """
    Load the train and validation data from the previously estimated poses.
    :param batch_size: batch size
    :type batch_size: int
    :param split_ratio: train and validation split ratio
    :type split_ratio: float
    :param path: path to the data
    :type path: str
    :return: train_loader, val_loader, train_coordinate_dataset, val_coordinate_dataset
    :rtype: torch.utils.data.dataloader.DataLoader,
            torch.utils.data.dataloader.DataLoader,
            CoordinatesDataset,
            CoordinatesDataset
    """
    numpy_data_world = load_pickle(path, "pose_world_landmark_numpy.pickle")
    labels_drop_na = load_pickle(path, "labels_drop_na.pickle")
    train_coordinate_dataset = CoordinatesDataset(numpy_data_world, labels_drop_na, set_type="train",
                                                  split_ratio=split_ratio)
    val_coordinate_dataset = CoordinatesDataset(numpy_data_world, labels_drop_na, set_type="val",
                                                split_ratio=split_ratio)

    # Create train and validation set
    train_loader = DataLoader(train_coordinate_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_coordinate_dataset, batch_size=batch_size, num_workers=12)

    return train_loader, val_loader, train_coordinate_dataset, val_coordinate_dataset
