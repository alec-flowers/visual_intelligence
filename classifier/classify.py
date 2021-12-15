import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from classifier.classifier_models import MLP
from classifier.test import evaluate_model
from classifier.train import train_model
from data.data import get_not_none_annotated_images, get_data
from pose.plot import plot_classified_images, plot_confusion_matrix
from pose.utils import MODEL_PATH, PICKLEDPATH


def classify_image(coordinates, reshape_inputs=True):
    if reshape_inputs:
        input = torch.from_numpy(coordinates)
        input = input.view(input.size(0), -1).float()

    # Set fixed random number seed
    torch.manual_seed(42)

    # Initialize the MLP
    mlp = MLP(num_classes=3)

    # Define the loss function and optimizer
    # loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    checkpoint = torch.load(str(MODEL_PATH) + "/mlp"+"/2021_11_22_20_46_21.ckpt")
    mlp.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    mlp.eval()
    with torch.no_grad():
        # Predict training data
        prediction = mlp(input)
        predicted_class = np.argmax(prediction, axis=1)
        return predicted_class


if __name__ == "__main__":
    # Define parameters
    save_plot = False
    train_from_scratch = True
    visualize_results = True
    min_valid_loss = np.inf
    split_ratio = 0.8

    # Model parameters
    batch_size = 64
    epochs = 100

    writer = SummaryWriter('runs/mlp/')

    train_loader, val_loader, train_coordinate_dataset, val_coordinate_dataset = \
        get_data(batch_size, split_ratio, PICKLEDPATH)

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
        train_model(mlp, train_loader, val_loader, loss_function, optimizer, epochs, writer, save_model=True)

    else:
        # Load a previously trained model
        model_version = "2021_12_10_10_42_37.ckpt"
        checkpoint = torch.load(str(MODEL_PATH) + "/mlp/" + model_version)
        mlp.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    annotated_images_filtered = get_not_none_annotated_images()
    annotated_images_filtered = \
        np.array(annotated_images_filtered, dtype=object)[train_coordinate_dataset.index_order]

    targets_train, predicted_class_train = evaluate_model(mlp, train_coordinate_dataset, reshape_inputs=True)
    targets_val, predicted_class_val = evaluate_model(mlp, val_coordinate_dataset, reshape_inputs=True)

    if visualize_results:
        # Confusion matrix
        plot_confusion_matrix(targets_val, predicted_class_val, 'validation', save_plot=save_plot)
        plot_confusion_matrix(targets_train, predicted_class_train, 'training', save_plot=save_plot)

        # Plot misclassified train images
        plot_classified_images(targets_train, predicted_class_train, annotated_images_filtered,
                               type_="training", max_n_to_plot=16, classified="misclassified", split_ratio=split_ratio)
        # Plot correctly train classified
        plot_classified_images(targets_train, predicted_class_train, annotated_images_filtered,
                               type_="training", max_n_to_plot=16, classified="correctly classified", split_ratio=split_ratio)

        # Plot misclassified test images
        plot_classified_images(targets_val, predicted_class_val, annotated_images_filtered,
                               type_="validation", max_n_to_plot=16, classified="misclassified", split_ratio=split_ratio)

        # Plot correctly test classified
        plot_classified_images(targets_val, predicted_class_val, annotated_images_filtered,
                               type_="validation", max_n_to_plot=16, classified="correctly classified",
                               split_ratio=split_ratio)

    # INSPECT training
    # run in terminal: tensorboard --logdir=runs
