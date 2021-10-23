from data import *
from utils import *
from plot import *
from pose import *

# Define parameters
shuffle = True

# Load the data
dataset, dataloader = load_data(path=TRAINPATH, batch_size=32, shuffle=shuffle)

# Visualize random data points with their labels
# plot_data(dataloader, n_images=4)

# Do the pose estimation
mp_pose = mp.solutions.pose  # available poses
estimated_poses, annotated_images = estimate_poses(dataloader)

plot_image_grid(annotated_images, n_images=4)
dataloader
