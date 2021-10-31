import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import Dataset, DataLoader, Subset
from utils import *
from PIL import Image
from collections import OrderedDict

from typing import Any, Callable, Optional, Tuple, List, Dict

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


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




def load_data(path=TRAINPATH, resize=300, batch_size=32, shuffle=True, batch_sampler=None, subset=False) -> Tuple[ClassifyDataset, DataLoader]:
    transform = (transforms.Compose([transforms.Resize(resize),
                                     transforms.CenterCrop(resize-1),
                                     #transforms.Grayscale(num_output_channels=3),
                                     transforms.ToTensor(),
                                     transforms.ConvertImageDtype(torch.uint8)]))

    dataset = ClassifyDataset(path, transform=transform)
    if subset:
        indices = torch.arange(20)
        dataset = Subset(dataset, indices)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, batch_sampler=batch_sampler)

    return dataset, dataloader
