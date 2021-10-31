from data import load_data
from utils import TRAINPATH, save_pickle, load_pickle
from plot import *
from pose import poses_for_dataset

if  __name__ == "__main__":
    # Define parameters
    shuffle = True

    # Load the data
    dataset, dataloader = load_data(path=TRAINPATH, batch_size=None, shuffle=shuffle, subset=True)

    # Do the pose estimation
    estimated_poses, annotated_images = poses_for_dataset(dataloader)
    a = estimated_poses[0]
    save_pickle(estimated_poses[0], "./data", "one_estimated_pose.pickle")