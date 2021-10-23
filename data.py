import torch
from torchvision import datasets, transforms
from utils import *


def load_data(path=TRAINPATH, resize=300, batch_size=32, shuffle=True):
    transform = (transforms.Compose([transforms.Resize(resize),
                                     transforms.CenterCrop(resize-1),
                                     #transforms.Grayscale(num_output_channels=3),
                                     transforms.ToTensor(),
                                     transforms.ConvertImageDtype(torch.uint8)]))

    dataset = datasets.ImageFolder(path, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataset, dataloader
