from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loading import create_angle_features
from pose.plot import plot_image_grid
from pose.pose_utils import PICKLEDPATH, load_pickle, BODY_POSE_CONNECTIONS


def front_leg(plane_x: np.array,
              plane_y: np.array,
              plane_z: np.array,
              left_foot: np.array,
              right_foot: np.array) -> str:
    """
    Given 3 points which form a plane, calculate a normal vector to the plane and then take the inner product to find
    which point is more on one side than the other.

    :param plane_x: 3D coordinate 1
    :param plane_y: 3D coordinate 2
    :param plane_z: 3D coordinate 3
    :param left_foot: 3D coordinate to evaluate which side of the plane it is on
    :param right_foot: 3D coordinate to evaluate which side of the plane it is on
    :return:
    """
    plane = np.array([plane_x, plane_y, plane_z])

    vect = plane[0:2] - plane[2]
    normal_vect = np.cross(vect[1], vect[0])
    left_foot_angle = np.dot(left_foot, normal_vect)
    right_foot_angle = np.dot(right_foot, normal_vect)

    if left_foot_angle > right_foot_angle:
        return 'LEFT'
    else:
        return 'RIGHT'


def which_leg_front(df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate which leg is in front for a single pose and change the angle names to FORWARD and BACKWARDS instead
    of RIGHT and LEFT.

    :param df_test: dataframe of keypoints and angles
    :return: dataframe of keypoints with renamed angles
    """
    df_test['forward'] = df_test.apply(
        lambda x: front_leg(x['LEFT_HIP'][0:3], x['RIGHT_HIP'][0:3], x['RIGHT_SHOULDER'][0:3],
                            x['LEFT_FOOT_INDEX'][0:3], x['RIGHT_FOOT_INDEX'][0:3]), axis=1)
    df_test = rename_columns(df_test, forward=df_test['forward'][0])
    # df_angle = df_test.iloc[:, 33:49]
    print(f"We calculated that the {df_test['forward'][0]} foot is the forward foot.")
    return df_test


def rename_columns(df: pd.DataFrame, forward: str = "LEFT") -> pd.DataFrame:
    """
    Rename the columns to FORWARD and BACKWARD instead of RIGHT and LEFT

    :param df: dataframe of keypoints and angles to rename
    :param forward: which foot is forward
    :return: dataframe with renamed columns
    """
    assert forward in ("LEFT","RIGHT")

    if forward == "LEFT":
        backward = "RIGHT"
    else:
        backward = "LEFT"

    new_name = []
    for i in df.columns:
        rep = i.replace(forward, "FORWARD").replace(backward, "BACKWARD")
        new_name.append(rep)

    rename_dict = dict(zip(df.columns, new_name))

    df = df.rename(rename_dict, axis=1)

    return df


def warrior_pose_front_back(df_pose: pd.DataFrame):
    """
    Calculate the front and back leg on an entire dataframe of poses and rename their columns.

    :param df_pose: df of poses
    :return: df pf poses with renamed columns
    """
    df_pose['forward'] = df_pose.apply(
        lambda x: front_leg(x['LEFT_HIP'][0:3], x['RIGHT_HIP'][0:3], x['RIGHT_SHOULDER'][0:3],
                            x['LEFT_FOOT_INDEX'][0:3], x['RIGHT_FOOT_INDEX'][0:3]), axis=1)

    df_pose_LEFT = df_pose[df_pose['forward'] == 'LEFT']
    df_pose_RIGHT = df_pose[df_pose['forward'] == 'RIGHT']
    df_pose_LEFT = rename_columns(df_pose_LEFT)
    df_pose_RIGHT = rename_columns(df_pose_RIGHT, forward="RIGHT")
    df_corrected = pd.concat([df_pose_LEFT, df_pose_RIGHT], axis=0)

    return df_corrected, df_corrected.columns[35:51]


def create_pose_df() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load dataframe of poses and split by pose.

    :return: split df by ground truth pose
    """
    df_world = load_pickle(PICKLEDPATH, "pose_world_landmark_all_df.pickle")
    df_world = df_world.replace(to_replace='None', value=np.nan).dropna()
    df_world = df_world.reset_index(drop=True)

    create_angle_features(df_world)

    df_w1 = df_world.loc[df_world['label'] == 1].copy()
    df_w2 = df_world.loc[df_world['label'] == 2].copy()
    df_dd = df_world.loc[df_world['label'] == 0].copy()

    return df_w1, df_w2, df_dd


def create_pose_np() -> pd.DataFrame:
    """
    Load in poses as a numpy array.

    :return: numpy array of poses
    """
    np_world = load_pickle(PICKLEDPATH, "pose_world_landmark_numpy.pickle")
    return np_world


def get_angle_confidence_intervals(df_pose: pd.DataFrame, angles: list, percent: float = .10) -> dict:
    """
    Calculate confidence intervals for a pose by looking at the distribution of angles.

    :param df_pose: dataframe of poses
    :param angles: names of the angles to do this calculation for
    :param percent: percent of angles to be outside of a tail
    :return: dict of angle names and their confidence intervals
    """
    df_pose = df_pose.loc[df_pose['quality'] == 1]
    num = df_pose.shape[0]
    num_to_take = int(num - num * (1 - percent))

    valid_angles = {}
    for idx, col in enumerate(angles):
        low = df_pose[col].sort_values(ascending=True).reset_index(drop=True)[num_to_take]
        median = df_pose[col].median()
        high = df_pose[col].sort_values(ascending=True).reset_index(drop=True)[num - num_to_take]

        valid_angles[col] = [low, median, high]

    return valid_angles


def calc_2d_angles(vec_2d: np.array) -> np.array:
    return np.degrees(np.arctan2(vec_2d[1], vec_2d[0]))


def calc_xyz_angles(vec: np.array) -> Tuple[np.array, np.array]:
    """
    Given a vector in 3D space. Take the (x,y) points and calculate the angle with the x axis. Take the (y, z) keypoints
    and calculate the angle with respect to the y axis.

    :param vec: point in 3D
    :return: angle on the xy plane and the angle on the yz plane
    """
    xy_points = vec[0:2]
    yz_points = vec[1:3]

    xy_angle = calc_2d_angles(xy_points)
    yz_angle = calc_2d_angles(yz_points)

    return xy_angle, yz_angle


def calc_xyz_dist(vec: np.array) -> Tuple[np.array, np.array]:
    """
    Given a vector in 3D space. Take the (x,y) points and calculate the length of the resulting 2D vector. Take the
    (y, z) keypoints and calculate the length with the resulting 2D vector.

    :param vec: point in 3D
    :return: length of xy vector, length of yz vector
    """
    xy_points = vec[0:2]
    yz_points = vec[1:3]

    xy_dist = np.linalg.norm(xy_points, axis=0)
    yz_dist = np.linalg.norm(yz_points, axis=0)

    return xy_dist, yz_dist


def calc_xyz_points(xy_angle: np.array, xy_dist: np.array, yz_angle: np.array, yz_dist: np.array) \
        -> Tuple[np.array, np.array, np.array]:
    """
    Calculate the location of a 3D point,

    :param xy_angle:
    :param xy_dist:
    :param yz_angle:
    :param yz_dist:
    :return: location of 3D point
    """
    x, y1 = calc_2d_points(xy_angle, xy_dist)
    y2, z = calc_2d_points(yz_angle, yz_dist)

    return x, y1, z


def calc_2d_points(angle: np.array, dist: np.array) -> Tuple[np.array, np.array]:
    """
    Given an angle and a length, calculate the location of the point in the 2D plane.

    :param angle:
    :param dist:
    :return: (x, y) of a 2D point
    """
    a = dist * np.cos(np.radians(angle))
    b = dist * np.sin(np.radians(angle))
    return a, b


def get_annotated_img(indx: int) -> np.array:
    """
    Load and select an certain annotated image

    :param indx: index of annotated image
    :return: image
    """
    annotated_images = load_pickle(PICKLEDPATH, "annotated_images.pickle")
    annotated_images = [img for img in annotated_images if img is not None]
    return annotated_images[indx]


def normalize_on_right_hip(keypoints: np.array) -> np.array:
    """
    Set the right hip as (0, 0, 0) and normalize the other keypoints.

    :param keypoints: keypoints to normalize
    :return: normalized keypoints
    """
    RIGHT_HIP = 24
    keypoints = keypoints - np.tile(np.swapaxes(keypoints[:, RIGHT_HIP, :][np.newaxis, :], 0, 1),
                                    (1, keypoints.shape[1], 1))
    return keypoints


def select_correct_closest_image(np_test: np.array, df: pd.DataFrame)\
        -> Tuple[np.array, int]:
    """
    Given a set of keypoints and a dataframe of poses. Find in the dataframe the pose that most closely matches the
    keypoints in a euclidian distance sense.

    :param np_test: set if keypoints for a certain pose
    :param df: dataframe of keypoints
    :return: keypoints of closest pose, index of closest pose
    """
    good_idx = df[df['quality'] == 1].index
    np_world = create_pose_np()
    np_good = np_world[good_idx]

    np_good = normalize_on_right_hip(np_good)
    np_test = normalize_on_right_hip(np_test)

    small_indx = np.argmin(np.sum(np.linalg.norm((np_good - np_test), axis=2), axis=1))
    ground_truth = np_good[small_indx, :, :]
    ground_truth_indx = good_idx[small_indx]

    return ground_truth, ground_truth_indx


def compare_two_figures(length_fig: np.array,
                        angle_fig: np.array,
                        img1: np.array,
                        img2: np.array,
                        plot: bool = True):
    """
    Take two figures and their keypoints. For one of them calculate the lengths from the right hip to all the keypoints.
    For the other calculate all the xy and yz angles from the right hip to the keypoints. Then combine these and
    create a new pose that keeps the lengths from the first pose, but manipulates it to be in the right angle from the
    second pose.

    :param length_fig: keypoints of the figure to calculate the lengths from
    :param angle_fig: keypoints of the figure to calculate the angles from
    :param img1: annotated image of length_fig
    :param img2: annotated image of angle_fig
    :param plot: display 3D plot
    """
    plot_image_grid([img1, img2], 2)

    xy_dist, yz_dist = calc_xyz_dist(length_fig.T)
    xy_angle, yz_angle = calc_xyz_angles(angle_fig.T)

    px, py, pz = calc_xyz_points(xy_angle, xy_dist, yz_angle, yz_dist)
    lx, ly, lz = length_fig.T[0], length_fig.T[1], length_fig.T[2]

    if plot:
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(px, py, pz)
        ax.scatter3D(lx, ly, lz)
        ax.view_init(elev=-50, azim=270)
        for i, j in BODY_POSE_CONNECTIONS:
            ax.plot([px[i], px[j]], [py[i], py[j]], [pz[i], pz[j]], color='b')
            ax.plot([lx[i], lx[j]], [ly[i], ly[j]], [lz[i], lz[j]], color='r')
        plt.show()
