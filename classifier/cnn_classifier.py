import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from classifier.classifier_models import CNN
from classifier.test import evaluate_model
from classifier.train import train_model
from data.data import load_data, train_val_split
from pose.plot import plot_confusion_matrix
from pose.utils import TRAINPATH, MODEL_PATH

if __name__ == "__main__":
    # Define parameters
    shuffle = True
    subset = False
    train_from_scratch = False
    save_model = train_from_scratch
    save_plot = False

    # Model parameters
    batch_size = 660
    epochs = 10
    gpus = 0

    # Load the data
    dataset, dataloader = load_data(path=TRAINPATH, resize=True, batch_size=batch_size, shuffle=False, subset=subset)
    images, labels = next(iter(dataloader))
    train_loader, val_loader, train_dataset, val_dataset = train_val_split(images, labels, batch_size=64,
                                                                           shuffle=shuffle, split_ratio=0.8)

    # plot_data(train_loader, 16)
    # plot_data(val_loader, 16)

    # Configure the training logging
    checkpoint_callback = ModelCheckpoint(monitor="train_loss", dirpath="saved_model/classifier/")
    writer = SummaryWriter('runs/classifier/')

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
        # Average training after epoch    10| Loss: 0.182 | Acc: 0.926
        # Average validation after epoch  10| Loss: 1.220 | Acc: 0.621
        checkpoint = torch.load(str(MODEL_PATH) + "/classifier" + "/2021_11_23_10_03_53.ckpt")
        cnn.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        targets_train, predicted_class_train = evaluate_model(cnn, train_dataset)
        targets_val, predicted_class_val = evaluate_model(cnn, val_dataset)

        plot_confusion_matrix(targets_val, predicted_class_val, 'validation', save_plot=save_plot)
        plot_confusion_matrix(targets_train, predicted_class_train, 'training', save_plot=save_plot)

    # INSPECT training
    # run in terminal: tensorboard --logdir=runs/classifier
