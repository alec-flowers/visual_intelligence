import datetime

import numpy as np
import torch

from utils import MODEL_PATH


def train_model(model, train_loader, val_loader, loss_function, optimizer, epochs, writer, save_model, mlp=True):

    min_valid_loss = np.inf

    # Run the training loop
    for epoch in range(0, epochs):
        # Set mlp in train mode
        model.train()

        # Print epoch
        print(f'Starting epoch {epoch + 1} / {epochs}')

        # Set current loss value
        current_loss = 0.0
        train_acc = 0.0
        # Iterate over the DataLoader for training data
        for i, data in enumerate(train_loader, 0):
            # Get inputs
            print(f"Processing batch {i}")
            inputs, targets = data

            # Transfer Data to GPU if available
            # if torch.cuda.is_available():
            #    inputs, targets = inputs.cuda(), targets.cuda()
            # Check if actually needed, since done in model
            if mlp:
                inputs = inputs.view(inputs.size(0), -1).float()

            # Zero the gradients
            optimizer.zero_grad()
            # Perform forward pass
            outputs = model(inputs)
            # Compute loss
            loss = loss_function(outputs, targets)
            # Perform backward pass
            loss.backward()
            # Perform optimization
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
        model.eval()  # Optional when not using Model Specific layer
        for inputs, targets in val_loader:
            # Transfer Data to GPU if available
            # if torch.cuda.is_available():
            #    inputs, targets = inputs.cuda(), targets.cuda()
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

        if save_model:
            if min_valid_loss > valid_loss:
                #  (from {min_valid_loss/len(val_loader):.6f} -> {valid_loss/len(val_loader):.6f})
                print(f'Validation Loss Decreased. Saving The Model...')
                min_valid_loss = valid_loss
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                           str(MODEL_PATH) + f"/model_intermediate.ckpt")
        # Print statistics after every epoch
        print('Average training after epoch   %3d| Loss: %.3f | Acc: %.3f ' %
              (epoch + 1, current_loss / len(train_loader), train_acc / len(train_loader.dataset)))
        print('Average validation after epoch %3d| Loss: %.3f | Acc: %.3f' %
              (epoch + 1, valid_loss / len(val_loader), valid_acc / len(val_loader.dataset)))

    print('Training process has finished.')

    if save_model:
        now = datetime.datetime.now()
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                   str(MODEL_PATH) + f"/{now.strftime('%Y_%m_%d_%H_%M_%S')}.ckpt")
