import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import Dataset, DataLoader, Subset
from utils import *
from PIL import Image
from collections import OrderedDict
import numpy as np

from typing import Any, Callable, Optional, Tuple, List, Dict

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class CoordinatesDataset(Dataset):
    def __init__(self, coordinates, labels, set_type, split_ratio=0.8):
        split_idx = int(split_ratio * coordinates.shape[0])
        if set_type == "train":
            self.coordinates = coordinates[:split_idx]
            self.labels = labels[:split_idx]
        elif set_type == "val":
            self.coordinates = coordinates[split_idx:]
            self.labels = labels[split_idx:]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        assert 0 <= idx < len(self)
        coordinates = self.coordinates[idx]
        label = self.labels[idx]

        return coordinates, label


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
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir() and not entry.name.startswith("2_"))
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        # Map classes to [downwardDog, warrior1, warrior2]
        class_to_idx = {cls_names: i % 3 for i, cls_names in enumerate(classes)}

        return classes, class_to_idx


def load_data(path=TRAINPATH, resize=False, batch_size=32, shuffle=True, batch_sampler=None, subset=False, subset_size=100) -> Tuple[ClassifyDataset, DataLoader]:
    if resize:
        resize_size = 300
        transform = (transforms.Compose([transforms.Resize(resize_size),
                                         transforms.CenterCrop(resize_size-1),
                                         #transforms.Grayscale(num_output_channels=3),
                                         transforms.ToTensor(),
                                         transforms.ConvertImageDtype(torch.uint8)]))
    else:
        transform = (transforms.Compose([# transforms.Grayscale(num_output_channels=3),
                                         transforms.ToTensor(),
                                         transforms.ConvertImageDtype(torch.uint8)]))
    dataset = ClassifyDataset(path, transform=transform)
    if subset:
        indices = torch.arange(subset_size)
        dataset = Subset(dataset, indices)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, batch_sampler=batch_sampler)

    return dataset, dataloader


def train_val_split(images, labels, batch_size=32, shuffle=True, split_ratio=0.8):
    train_dataset = RawImageDataset(images, labels, set_type="train", split_ratio=split_ratio)
    val_dataset = RawImageDataset(images, labels, set_type="val", split_ratio=split_ratio)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=None, shuffle=shuffle)
    return train_dataloader, val_loader


class RawImageDataset(Dataset):
    def __init__(self, images, labels, set_type, split_ratio=0.8):
        split_idx = int(split_ratio * images.shape[0])
        if set_type == "train":
            self.images = images[:split_idx]
            self.labels = labels[:split_idx]
        elif set_type == "val":
            self.images = images[split_idx:]
            self.labels = labels[split_idx:]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        assert 0 <= idx < len(self)
        images = self.images[idx]
        label = self.labels[idx]

        return images, label
