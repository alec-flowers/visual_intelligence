import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from data import load_data, train_val_split
from model import CNN
from train import train_model
from utils import TRAINPATH

if __name__ == "__main__":
    # Define parameters
    shuffle = True
    subset = True
    save_model = False
    # Model parameters
    batch_size = 660
    epochs = 5
    gpus = 0

    # Load the data
    dataset, dataloader = load_data(path=TRAINPATH, resize=True, batch_size=batch_size, shuffle=shuffle, subset=subset)
    images, labels = next(iter(dataloader))
    train_loader, val_loader = train_val_split(images, labels, batch_size=32, shuffle=True, split_ratio=0.8)

    # Configure the training logging
    logger = TensorBoardLogger("tb_logs", name="cnn")
    checkpoint_callback = ModelCheckpoint(monitor="train_loss", dirpath="saved_model/cnn/")
    writer = SummaryWriter('runs/cnn/')

    # Set fixed random number seed
    torch.manual_seed(42)

    # Initialize the CNN
    cnn = CNN(num_classes=3)

    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)

    train_model(cnn, train_loader, val_loader, loss_function, optimizer, epochs, writer, save_model)

    cnn
    # INSPECT training
    # run in terminal: tensorboard --logdir=runs/cnn
