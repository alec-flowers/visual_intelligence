import pytorch_lightning as pl
import pandas as pd
import cv2
import os

from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import Dataset ,DataLoader
import numpy as np
import torch
from data import load_data, train_val_split
from model import CNN
from utils import TRAINPATH


if __name__ == "__main__":
    # Define parameters
    shuffle = True
    subset = True
    # Model parameters
    batch_size = 660
    epochs = 5
    gpus = 0

    # Load the data
    dataset, dataloader = load_data(path=TRAINPATH, resize=True, batch_size=batch_size, shuffle=shuffle, subset=subset)
    images, labels = next(iter(dataloader))
    train_dataloader, val_loader = train_val_split(images, labels, batch_size=32, shuffle=True, split_ratio=0.8)

    # Init the model
    pl.seed_everything(42)
    cnn = CNN(num_classes=3)
    logger = TensorBoardLogger("tb_logs", name="cnn")
    # Configure the training
    trainer = pl.Trainer(gpus=gpus, deterministic=True, log_every_n_steps=5, max_epochs=epochs, logger=logger)
    # Start training
    trainer.fit(cnn, train_dataloader)
    cnn