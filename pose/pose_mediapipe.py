import argparse
import os
import sys
from typing import Tuple, Any, NamedTuple

import cv2
import mediapipe as mp
import mediapipe.python.solution_base
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data.data_loading import load_data, ClassifyDataset
from pose.plot import plot_image, plot_dataset_images, plot_annotated_images, plot_no_pose_photo
from pose.pose_utils import LANDMARK_DICT, CLASS_MAPPINGS_IDX, POSE_QUALITY_MAPPINGS, TRAINPATH
from pose.pose_utils import PICKLEDPATH, save_dataframes_to_pickle, \
    POSEDATAFRAME_LIST

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def poses_for_dataset(dataloader: torch.utils.data.dataloader.DataLoader,
                      skip_image_annotation:
                      bool = False) \
        -> Tuple[list, list]:
    assert dataloader.batch_size is None
    result_list = []
    annotated_images = []
    for image, label in tqdm(dataloader):
        results, annotated = estimate_poses(image, label, skip_image_annotation)
        result_list.append(results)
        annotated_images.append(annotated)
    return result_list, annotated_images


def estimate_poses(image: torch.Tensor,
                   label: torch.Tensor,
                   skip_image_annotation: bool,
                   plot: bool = False) \
        -> Tuple[NamedTuple, Any]:
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


def pose_landmarks_to_list(solution: mediapipe.python.solution_base.SolutionBase,
                           pose_var: str) \
        -> Tuple[list, list, list]:
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


def pose_to_dataframe(estimated_poses: list,
                      dataset: ClassifyDataset,
                      pose_var: str,
                      good_bad: bool = False) \
        -> Tuple[pd.DataFrame, pd.DataFrame, np.array, np.array]:
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

    if good_bad:
        df['label'] = df.apply(lambda x: POSE_QUALITY_MAPPINGS[x['tmp']], axis=1)
    else:
        df['label'] = df.apply(lambda x: CLASS_MAPPINGS_IDX[x['tmp']], axis=1)

    df['quality'] = df.apply(lambda x: POSE_QUALITY_MAPPINGS[x['tmp']], axis=1)
    df.drop('tmp', axis=1, inplace=True)

    if good_bad:
        df_vis['label'] = df_vis.apply(lambda x: POSE_QUALITY_MAPPINGS[x['tmp']], axis=1)
    else:
        df_vis['label'] = df_vis.apply(lambda x: CLASS_MAPPINGS_IDX[x['tmp']], axis=1)

    df_vis['quality'] = df_vis.apply(lambda x: POSE_QUALITY_MAPPINGS[x['tmp']], axis=1)
    df_vis.drop('tmp', axis=1, inplace=True)

    labels_drop_na = np.array(df_vis.dropna(axis=0, how='any')['label'])

    return df, df_vis, np.array(for_numpy), labels_drop_na


def parse_args():
    parser = argparse.ArgumentParser(description='Pose estimation.')
    parser.add_argument("-path", type=str,
                        required=False, default=TRAINPATH,
                        help="Path to the images where you want to extract poses from.")
    parser.add_argument("-save", type=str,
                        required=False, default=PICKLEDPATH,
                        help="Path to save the estimated poses and extracted dataframes to.")
    parser.add_argument("-viz", type=bool,
                        required=False, default=True,
                        help="Visualize some of the estimated poses.")
    parser.add_argument("-skip", type=bool,
                        required=False, default=False,
                        help="Skip saving the annotated images, because they take a lot of memory.")
    parser.add_argument("-gb", type=bool,
                        required=False, default=False,
                        help="Whether we want to distinguish between good and bad poses or not.")
    return parser.parse_args()


def main(args):
    dataset, dataloader = load_data(path=args.path, batch_size=None, shuffle=False, good_bad=args.gb)

    # Pose estimation
    estimated_poses, annotated_images = poses_for_dataset(dataloader, skip_image_annotation=args.skip)

    df, df_vis, numpy_data, labels_drop_na = \
        pose_to_dataframe(estimated_poses, dataset, pose_var='pose_landmarks', good_bad=args.gb)
    df_world, df_vis_world, numpy_data_world, _ = \
        pose_to_dataframe(estimated_poses, dataset, pose_var='pose_world_landmarks', good_bad=args.gb)

    save_dataframes_to_pickle(args.save,
                              [df, df_vis, numpy_data, df_world, df_vis_world, numpy_data_world,
                               labels_drop_na, annotated_images],
                              POSEDATAFRAME_LIST
                              )

    if args.viz:
        plot_dataset_images(dataset, 9)
        plot_no_pose_photo(df, dataset, 9)
        if not args.skip:
            plot_annotated_images(annotated_images, 9)


if __name__ == "__main__":
    args = parse_args()
    main(args)
