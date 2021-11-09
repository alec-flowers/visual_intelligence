import datetime

import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader

from data import CoordinatesDataset
# from plot import *
from model import MLP
from plot import plot_image_grid
from utils import load_pickle, PICKLEDPATH, MODEL_PATH

if __name__ == "__main__":
    # Define parameters
    shuffle = True
    # load_from_scratch = True
    train_from_scratch = False
    save_model = False
    min_valid_loss = np.inf

    # Model parameters
    batch_size = 64
    epochs = 50

    # Load the data
    numpy_data_world = load_pickle(PICKLEDPATH, "pose_world_landmark_numpy.pickle")
    labels_drop_na = load_pickle(PICKLEDPATH, "labels_drop_na.pickle")
    train_coordinate_dataset = CoordinatesDataset(numpy_data_world, labels_drop_na, set_type="train")
    val_coordinate_dataset = CoordinatesDataset(numpy_data_world, labels_drop_na, set_type="val")
    # Create train and validation set
    train_loader = DataLoader(train_coordinate_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_coordinate_dataset, batch_size=batch_size, num_workers=12)

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
        # Run the training loop
        for epoch in range(0, epochs):
            # Set mlp in train mode
            mlp.train()

            # Print epoch
            print(f'Starting epoch {epoch + 1}')

            # Set current loss value
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(train_loader, 0):
                # Get inputs
                inputs, targets = data

                # Transfer Data to GPU if available
                #if torch.cuda.is_available():
                #    inputs, targets = inputs.cuda(), targets.cuda()
                inputs = inputs.view(inputs.size(0), -1).float()

                # Zero the gradients
                optimizer.zero_grad()
                # Perform forward pass
                outputs = mlp(inputs)
                # Compute loss
                loss = loss_function(outputs, targets)
                # Perform backward pass
                loss.backward()
                # Perform optimization
                optimizer.step()
                current_loss += loss.item()

            # Iterate over the DataLoader for validation data
            valid_loss = 0.0
            mlp.eval()  # Optional when not using Model Specific layer
            for inputs, targets in val_loader:
                # Transfer Data to GPU if available
                #if torch.cuda.is_available():
                #    inputs, targets = inputs.cuda(), targets.cuda()
                inputs = inputs.view(inputs.size(0), -1).float()
                outputs = mlp(inputs)
                loss = loss_function(outputs, targets)
                valid_loss += loss.item()

            if save_model:
                if min_valid_loss > valid_loss:
                    print(f'Validation Loss Decreased (from {min_valid_loss:.6f} -> {valid_loss:.6f}). Saving The Model...')
                    min_valid_loss = valid_loss
                    torch.save({'mlp_state_dict': mlp.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                               str(MODEL_PATH) + f"/mlp_intermediate.ckpt")
            # Print statistics after every epoch
            # if (epoch-1) % 2 == 0:  # Update every 2 epochs
            print('Loss after epoch %3d: %.3f' %
                  (epoch + 1, current_loss))

        # Process is complete.
        print('Training process has finished.')

        if save_model:
            now = datetime.datetime.now()
            torch.save({'mlp_state_dict': mlp.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                       str(MODEL_PATH) + f"/{now.strftime('%Y_%m_%d_%H_%M_%S')}.ckpt")

    else:
        checkpoint = torch.load(str(MODEL_PATH) + "/2021_11_09_20_14_20.ckpt")
        mlp.load_state_dict(checkpoint['mlp_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Evaluate the model
    mlp.eval()

    with torch.no_grad():
        inputs = torch.from_numpy(train_coordinate_dataset.coordinates)
        inputs = inputs.view(inputs.size(0), -1).float()
        targets = torch.from_numpy(train_coordinate_dataset.labels)
        predictions = mlp(inputs)
        predicted_class = np.argmax(predictions, axis=1)
        plot_image_grid(images, n_images, dataloader=False, title="")
    mlp

    # TODO visualize coordinates together with labels to verify they are aligned
    # TODO set up tensorboard again
    # INSPECT training
    # run in terminal: tensorboard --logdir=tb_logs
