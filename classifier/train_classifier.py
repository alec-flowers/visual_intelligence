import datetime

import numpy as np
import torch
from torch import nn

from pose.pose_utils import MODEL_PATH


def train_model(model,
                train_loader: torch.utils.data.dataloader.DataLoader,
                val_loader: torch.utils.data.dataloader.DataLoader,
                loss_function: nn.CrossEntropyLoss,
                optimizer: torch.optim.Adam,
                epochs: int,
                writer: torch.utils.tensorboard.writer.SummaryWriter,
                model_path: str = MODEL_PATH,
                mlp: bool = True):
    """
    Train an MLP or CNN model with some data and validate the training process after each epoch.

    :param model: A model from our model.py file
    :type model: MLP or CNN
    :param train_loader: the training data
    :type train_loader: torch.utils.data.dataloader.DataLoader
    :param val_loader: the validation data
    :type val_loader: torch.utils.data.dataloader.DataLoader
    :param loss_function: CrossEntropyLoss
    :type loss_function: nn.CrossEntropyLoss
    :param optimizer: Adam
    :type optimizer: torch.optim.Adam
    :param epochs: epochs to train
    :type epochs: int
    :param writer: log the training process
    :type writer: torch.utils.tensorboard.writer.SummaryWriter
    :param model_path: path to save the trained model
    :type model_path: str
    :param mlp: whether we train the MLP or the CNN
    :type mlp: bool
    """
    min_valid_loss = np.inf
    for epoch in range(0, epochs):
        model.train()
        print(f'Starting epoch {epoch + 1} / {epochs}')
        current_loss = 0.0
        train_acc = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(train_loader, 0):
            print(f"Processing batch {i}")
            inputs, targets, _ = data
            if mlp:
                inputs = inputs.view(inputs.size(0), -1).float()

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()

            # Get accuracy
            pred = torch.argmax(outputs, dim=1)
            train_acc += torch.sum(pred == targets)

        # Log the average loss after an epoch
        writer.add_scalar('training loss',
                          current_loss / len(train_loader),
                          epoch)
        writer.add_scalar('training accuracy',
                          train_acc / len(train_loader.dataset),
                          epoch)

        # Iterate over the DataLoader for validation data
        valid_loss = 0.0
        valid_acc = 0.0
        model.eval()
        for inputs, targets, _ in val_loader:
            if mlp:
                inputs = inputs.view(inputs.size(0), -1).float()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            valid_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            valid_acc += torch.sum(pred == targets)

        # Log the average validation loss after an epoch
        writer.add_scalar('validation loss',
                          valid_loss / len(val_loader),
                          epoch)
        writer.add_scalar('validation accuracy',
                          valid_acc / len(val_loader.dataset),
                          epoch)

        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased. Saving The Model...')
            min_valid_loss = valid_loss
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                       str(model_path) + f"/model_intermediate.ckpt")

        # Print statistics after every epoch
        print('Average training after epoch   %3d| Loss: %.3f | Acc: %.3f ' %
              (epoch + 1, current_loss / len(train_loader), train_acc / len(train_loader.dataset)))
        print('Average validation after epoch %3d| Loss: %.3f | Acc: %.3f' %
              (epoch + 1, valid_loss / len(val_loader), valid_acc / len(val_loader.dataset)))
    print('Training process has finished.')

    now = datetime.datetime.now()
    model_type = '/mlp' if mlp else '/classifier'
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
               str(model_path) + model_type+f"/{now.strftime('%Y_%m_%d_%H_%M_%S')}.ckpt")
