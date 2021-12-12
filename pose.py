import os
import sys

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from data import load_data
from plot import *
from utils import LANDMARK_DICT, load_pickle, CLASS_MAPPINGS_IDX, POSE_QUALITY_MAPPINGS
from utils import TRAINPATH, PICKLEDPATH, save_dataframes_to_pickle, \
    POSEDATAFRAME_LIST

# from plot import *


module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def poses_for_dataset(dataloader, skip_image_annotation: bool = False):
    assert dataloader.batch_size is None
    result_list = []
    annotated_images = []
    for image, label in tqdm(dataloader):
        results, annotated = estimate_poses(image, label, skip_image_annotation)
        result_list.append(results)
        annotated_images.append(annotated)
    return result_list, annotated_images


def estimate_poses(image, label, skip_image_annotation, plot=False):
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.0) as pose:
        image = image.numpy().transpose((1, 2, 0))
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            if plot:
                plot_image(image, label.item(), "Landmark not available")
            return None, None

        if skip_image_annotation:
            return results, None

        annotated_image = image.copy()
        # Draw pose landmarks on the image.
        # Set visibility of all landmarks to 1 for plotting
        for elem in results.pose_landmarks.landmark:
            elem.visibility = 1.0

        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # Plot pose world landmarks.
        if plot:
            plot_image(annotated_image)

    return results, annotated_image


def pose_landmarks_to_list(solution, pose_var):
    val = []
    visib = []
    nump = []
    for land in getattr(solution, pose_var).landmark:
        x = land.x
        y = land.y
        z = land.z
        vis = land.visibility
        val.append([x, y, z, vis])
        nump.append([x, y, z])
        visib.append(vis)

    return val, visib, nump


def pose_to_dataframe(estimated_poses, dataset, pose_var):
    all_val = []
    visib_val = []
    for_numpy = []
    for i in range(len(estimated_poses)):
        if estimated_poses[i]:
            val, visib, nump = pose_landmarks_to_list(estimated_poses[i], pose_var)
            all_val.append(val)
            visib_val.append(visib)
            for_numpy.append(nump)
        else:
            all_val.append([None])
            visib_val.append([None])

    df = pd.DataFrame.from_records(all_val)
    df_vis = pd.DataFrame.from_records(visib_val)

    df = df.rename(LANDMARK_DICT, axis=1)
    df_vis = df_vis.rename(LANDMARK_DICT, axis=1)

    labels = []
    for _, label in dataset:
        labels.append(label)
    df['tmp'] = labels
    df_vis['tmp'] = labels

    df['label'] = df.apply(lambda x: CLASS_MAPPINGS_IDX[x['tmp']], axis=1)
    df['quality'] = df.apply(lambda x: POSE_QUALITY_MAPPINGS[x['tmp']], axis=1)
    df.drop('tmp', axis=1, inplace=True)

    df_vis['label'] = df_vis.apply(lambda x: CLASS_MAPPINGS_IDX[x['tmp']], axis=1)
    df_vis['quality'] = df_vis.apply(lambda x: POSE_QUALITY_MAPPINGS[x['tmp']], axis=1)
    df_vis.drop('tmp', axis=1, inplace=True)

    labels_drop_na = np.array(df_vis.dropna(axis=0, how='any')['label'])

    return df, df_vis, np.array(for_numpy), labels_drop_na


if __name__ == "__main__":
    # Define parameters
    shuffle = False
    run_from_scratch = False
    subset = False  # Take a subset of 100 images out of the 660 images?
    save_poses = True  # Save poses after estimated?

    # Load the data
    dataset, dataloader = load_data(path=TRAINPATH, batch_size=None, shuffle=shuffle, subset=subset, subset_size=100)

    if run_from_scratch:
        # Do the pose estimation
        estimated_poses, annotated_images = poses_for_dataset(dataloader, skip_image_annotation=True)
        # plot_annotated_images(annotated_images, 9)

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
        df = load_pickle(PICKLEDPATH, "pose_landmark_all_df.pickle")
        df_vis = load_pickle(PICKLEDPATH, "pose_landmark_vis_df.pickle")
        numpy_data = load_pickle(PICKLEDPATH, "pose_landmark_numpy.pickle")
        df_world = load_pickle(PICKLEDPATH, "pose_world_landmark_all_df.pickle")
        df_vis_world = load_pickle(PICKLEDPATH, "pose_world_landmark_vis_df.pickle")

        annotated_images = load_pickle(PICKLEDPATH, "annotated_images.pickle")

        # plot_dataset_images(dataset, 9)
        # plot_annotated_images(annotated_images, 16)
        # plot_no_pose_photo(df, dataset, 9)

        print(f"There are {sum(df['NOSE'].isna())} images we don't get a pose estimate for out \
        of {len(dataset)}. This is {sum(df['NOSE'].isna()) / len(dataset) * 100:.2f}%")

        df_vis = df_vis.dropna(axis=0, how='any')
        print(df_vis.describe())
        df_vis.mean().sort_values(ascending=False)
