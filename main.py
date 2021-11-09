from data import *
from utils import *
from plot import *
from pose import *

if __name__ == "__main__":
    # Define parameters
    shuffle = True

    # Load the data
    dataset, dataloader = load_data(path=TRAINPATH, batch_size=32, shuffle=shuffle, subset=True, subset_size=16)

    # Visualize random data points with their labels
    # TODO doesnt work anymore
    plot_data(dataloader, n_images=4)

    # Do the pose estimation
    # TODO doesnt work anymore
    mp_pose = mp.solutions.pose  # available poses
    estimated_poses, annotated_images = estimate_poses(dataloader)

    plot_image_grid(annotated_images, n_images=4)
