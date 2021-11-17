import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from data import get_not_none_annotated_images, get_data
from model import MLP
from test import evaluate_mlp
from train import train_model
from utils import MODEL_PATH

if __name__ == "__main__":
    # Define parameters
    shuffle = True
    train_from_scratch = False
    save_model = False
    min_valid_loss = np.inf
    split_ratio = 0.8

    # Model parameters
    batch_size = 64
    epochs = 25

    writer = SummaryWriter('runs/')
    # TODO we have to shuffle the data somehow, bc rn we have only dd in our validation data ...
    # Maybe shuffle but keep the indices such that for the annotated images we now how to sort them?
    # Load the data
    train_loader, val_loader, train_coordinate_dataset, val_coordinate_dataset = get_data(batch_size, split_ratio)
    # Configure the training logging
    logger = TensorBoardLogger("tb_logs", name="mlp")
    checkpoint_callback = ModelCheckpoint(monitor="train_loss", dirpath="saved_model/mlp/")

    # Set fixed random number seed
    torch.manual_seed(42)

    # Initialize the MLP
    mlp = MLP(num_classes=3)

    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    if train_from_scratch:
        train_model(mlp, train_loader, val_loader, loss_function, optimizer, epochs, writer, save_model)

    else:
        # Load a previously trained model that has the following stats:
        # Average training after epoch  25  | Loss: 0.411 | Acc: 0.918
        # Average validation after epoch  25| Loss: 0.230 | Acc: 0.953
        checkpoint = torch.load(str(MODEL_PATH) + "/2021_11_17_06_40_29.ckpt")
        mlp.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        annotated_images_filtered = get_not_none_annotated_images()

        # Plot misclassified train images
        evaluate_mlp(mlp, train_coordinate_dataset, annotated_images_filtered, plot="misclassified", type_="training")
        # Plot correctly train classified
        evaluate_mlp(mlp, train_coordinate_dataset, annotated_images_filtered, plot="correctly classified", type_="training")
        # Plot misclassified test images
        evaluate_mlp(mlp, val_coordinate_dataset, annotated_images_filtered,
                     plot="misclassified", type_="validation", split_ratio=split_ratio)
        # Plot correctly test classified
        evaluate_mlp(mlp, val_coordinate_dataset, annotated_images_filtered,
                     plot="correctly classified", type_="validation", split_ratio=split_ratio)

    mlp

    # INSPECT training
    # run in terminal: tensorboard --logdir=runs
