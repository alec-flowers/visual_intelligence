import argparse

import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from classifier.classifier_models import MLP
from classifier.test_classifier import evaluate_model
from classifier.train_classifier import train_model
from data.data_loading import get_not_none_annotated_images, get_data
from pose.plot import plot_classified_images, plot_confusion_matrix
from pose.pose_utils import MODEL_PATH, PICKLEDPATH


def classify_image(coordinates: np.array, reshape_inputs: bool = True) -> np.array:
    if reshape_inputs:
        input = torch.from_numpy(coordinates)
        input = input.view(input.size(0), -1).float()

    torch.manual_seed(42)

    mlp = MLP(num_classes=3)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    checkpoint = torch.load(str(MODEL_PATH) + "/mlp"+"/2021_11_22_20_46_21.ckpt")
    mlp.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    mlp.eval()
    with torch.no_grad():
        prediction = mlp(input)
        predicted_class = np.argmax(prediction, axis=1)
        return predicted_class


def parse_args():
    parser = argparse.ArgumentParser(description='Pose classification.')
    parser.add_argument("-pickles", type=str,
                        required=False, default=PICKLEDPATH,
                        help="Path to load the pickled dataframes from the pose estimation from.")
    parser.add_argument("-scratch", type=bool, default=False,
                        help="Train the classifier from scratch or load a previously trained model.")
    parser.add_argument("-version", type=str,
                        required=False, default="2021_12_10_10_42_37.ckpt",
                        help="If you don't train from scratch, specify the model version to be resumed for training.")
    parser.add_argument("-save", type=str,
                        required=False, default=MODEL_PATH,
                        help="Path to save and load the trained model to/ from.")
    parser.add_argument("-epochs", type=int,
                        required=False, default=100,
                        help="How many epochs to train the model for.")
    parser.add_argument("-viz", type=bool,
                        required=False, default=True,
                        help="Visualize confusion matrices of the classifier and "
                             "correctly and incorrectly classified images.")

    return parser.parse_args()


def main(args):
    # Define parameters
    split_ratio = 0.8
    batch_size = 64

    writer = SummaryWriter('runs/mlp/')

    train_loader, val_loader, train_coordinate_dataset, val_coordinate_dataset = \
        get_data(batch_size, split_ratio, args.pickles)

    # Configure the training logging
    logger = TensorBoardLogger("tb_logs", name="mlp")
    checkpoint_callback = ModelCheckpoint(monitor="train_loss", dirpath="saved_model/mlp/")

    torch.manual_seed(42)

    mlp = MLP(num_classes=3)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    if args.scratch:
        train_model(mlp, train_loader, val_loader, loss_function, optimizer, args.epochs, writer, args.save)

    else:
        # Load a previously trained model
        checkpoint = torch.load(str(args.save) + "/mlp/" + args.version)
        mlp.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    annotated_images_filtered = get_not_none_annotated_images(args.pickles)
    if annotated_images_filtered:
        annotated_images_filtered = \
            np.array(annotated_images_filtered, dtype=object)[train_coordinate_dataset.index_order]

    targets_train, predicted_class_train = evaluate_model(mlp, train_coordinate_dataset, reshape_inputs=True)
    targets_val, predicted_class_val = evaluate_model(mlp, val_coordinate_dataset, reshape_inputs=True)

    if args.viz:
        # Confusion matrix
        plot_confusion_matrix(targets_val, predicted_class_val, 'validation', save_plot=False)
        plot_confusion_matrix(targets_train, predicted_class_train, 'training', save_plot=False)

        if annotated_images_filtered is not None:
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


if __name__ == "__main__":
    args = parse_args()
    main(args)

