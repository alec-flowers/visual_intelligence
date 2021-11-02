from data import load_data, CoordinatesDataset
from utils import TRAINPATH, save_pickle, load_pickle, DATAPATH, PICKLEDPATH, save_dataframes_to_pickle, \
    POSEDATAFRAME_LIST
# from plot import *
from pose import poses_for_dataset, pose_to_dataframe  # , plot_dataset_images, plot_annotated_images, plot_no_pose_photo
from model import MLP
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger


if __name__ == "__main__":
    # Define parameters
    shuffle = True

    # Model parameters
    batch_size = 32
    epochs = 5
    gpus = 0

    # Load the data
    numpy_data_world = load_pickle(PICKLEDPATH, "pose_world_landmark_numpy.pickle")
    labels_drop_na = load_pickle(PICKLEDPATH, "labels_drop_na.pickle")
    train_coordinate_dataset = CoordinatesDataset(numpy_data_world, labels_drop_na, set_type="train")
    val_coordinate_dataset = CoordinatesDataset(numpy_data_world, labels_drop_na, set_type="val")
    train_dataloader = DataLoader(train_coordinate_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    val_dataloader = DataLoader(val_coordinate_dataset, batch_size=batch_size, num_workers=12)

    # Init the model
    pl.seed_everything(42)
    mlp = MLP(num_classes=3)
    logger = TensorBoardLogger("tb_logs", name="mlp")
    # Configure the training
    trainer = pl.Trainer(gpus=gpus, deterministic=True, log_every_n_steps=5, max_epochs=epochs, logger=logger)
    # Start training
    trainer.fit(mlp, train_dataloader)

    # TODO how to include validation
    #trainer.fit(mlp, train_dataloader, val_dataloader)
    mlp
    # TODO visualize coordinates together with labels to verify they are aligned

    # INSPECT training
    # run in terminal: tensorboard --logdir=tb_logs
