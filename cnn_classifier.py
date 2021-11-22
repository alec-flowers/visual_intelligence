import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from data import load_data, train_val_split, get_not_none_annotated_images
from model import CNN
from train import train_model
from utils import TRAINPATH, MODEL_PATH

if __name__ == "__main__":
    # Define parameters
    shuffle = True
    subset = True
    train_from_scratch = False
    save_model = train_from_scratch

    # Model parameters
    batch_size = 660
    epochs = 10
    gpus = 0

    # Load the data
    dataset, dataloader = load_data(path=TRAINPATH, resize=True, batch_size=batch_size, shuffle=shuffle, subset=subset)
    images, labels = next(iter(dataloader))
    train_loader, val_loader = train_val_split(images, labels, batch_size=32, shuffle=True, split_ratio=0.8)

    # plot_data(train_loader, 16)
    # plot_data(val_loader, 16)

    # Configure the training logging
    logger = TensorBoardLogger("tb_logs", name="cnn")
    checkpoint_callback = ModelCheckpoint(monitor="train_loss", dirpath="saved_model/cnn/")
    writer = SummaryWriter('runs/cnn/')

    # Set fixed random number seed
    torch.manual_seed(17)

    # Initialize the CNN
    cnn = CNN(num_classes=3)

    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)

    if train_from_scratch:
        train_model(cnn, train_loader, val_loader, loss_function, optimizer, epochs, writer, save_model, mlp=False)
    else:
        # Load a previously trained model that has the following stats:
        # Average training after epoch    10| Loss: 0.000 | Acc: 1.000
        # Average validation after epoch  10| Loss: 3.043 | Acc: 0.950
        checkpoint = torch.load(str(MODEL_PATH) + "/cnn" + "/2021_11_22_23_07_41.ckpt")
        cnn.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        annotated_images_filtered = get_not_none_annotated_images()
        # if shuffle:
        #   annotated_images_filtered = np.array(annotated_images_filtered, dtype=object)[train_coordinate_dataset.index_order]

    cnn

    # INSPECT training
    # run in terminal: tensorboard --logdir=runs/cnn
