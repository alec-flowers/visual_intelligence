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
    run_from_scratch = False
    subset = False  # Take a subset of 100 images out of the 660 images?
    save_poses = False  # Save poses after estimated?

    # Model parameters
    batch_size = 32
    epochs = 5
    gpus = 0

    # Load the data
    dataset, dataloader = load_data(path=TRAINPATH, batch_size=None, shuffle=shuffle, subset=subset)

    # TODO: I suspect the annotated images take lots of memory, for me (Nina) there was an issue when loading them
    #  from the pickle
    if run_from_scratch:
        # Do the pose estimation
        estimated_poses, annotated_images = poses_for_dataset(dataloader)

        # NOTE NUMPY DATA TAKES OUT NULLS, will have to take out nulls in labels
        df, df_vis, numpy_data, labels_drop_na = pose_to_dataframe(estimated_poses, dataset, pose_var='pose_landmarks')
        df_world, df_vis_world, numpy_data_world, _ = pose_to_dataframe(estimated_poses, dataset,
                                                                        pose_var='pose_world_landmarks')
        if save_poses:
            save_dataframes_to_pickle(PICKLEDPATH,
                                      [df, df_vis, numpy_data, df_world, df_vis_world, numpy_data_world,
                                       labels_drop_na, annotated_images],
                                      POSEDATAFRAME_LIST
                                      )

    else:
        # df = load_pickle(PICKLEDPATH, "pose_landmark_all_df.pickle")
        # df_vis = load_pickle(PICKLEDPATH, "pose_landmark_vis_df.pickle")
        # numpy_data = load_pickle(PICKLEDPATH, "pose_landmark_numpy.pickle")
        # df_world = load_pickle(PICKLEDPATH, "pose_world_landmark_all_df.pickle")
        # df_vis_world = load_pickle(PICKLEDPATH, "pose_world_landmark_vis_df.pickle")

        # annotated_images = load_pickle(PICKLEDPATH, "annotated_images.pickle") # TODO doesnt work for me (Nina)

        # plot_dataset_images(dataset, 9)
        # plot_annotated_images(annotated_images, 16)  # TODO doesnt work
        # plot_no_pose_photo(df, dataset, 9)

        # print(f"There are {sum(df['NOSE'].isna())} images we don't get a pose estimate for out \
        # of {len(dataset)}. This is {sum(df['NOSE'].isna()) / len(dataset) * 100:.2f}%")

        # df_vis = df_vis.dropna(axis=0, how='any')
        # df_vis.describe()
        # df_vis.mean().sort_values(ascending=False)

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
