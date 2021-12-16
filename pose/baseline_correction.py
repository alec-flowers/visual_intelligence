import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loading import create_angle_features
from pose.plot import plot_image_grid
from pose.pose_utils import PICKLEDPATH, load_pickle, BODY_POSE_CONNECTIONS


def front_leg(plane_x, plane_y, plane_z, left_foot, right_foot):
    plane = np.array([plane_x, plane_y, plane_z])

    vect = plane[0:2] - plane[2]
    normal_vect = np.cross(vect[1], vect[0])
    left_foot_angle = np.dot(left_foot, normal_vect)
    right_foot_angle = np.dot(right_foot, normal_vect)

    if left_foot_angle > right_foot_angle:
        return 'LEFT'
    else:
        return 'RIGHT'


def which_leg_front(df_test):
    df_test['forward'] = df_test.apply(
        lambda x: front_leg(x['LEFT_HIP'][0:3], x['RIGHT_HIP'][0:3], x['RIGHT_SHOULDER'][0:3],
                            x['LEFT_FOOT_INDEX'][0:3], x['RIGHT_FOOT_INDEX'][0:3]), axis=1)
    df_test = rename_columns(df_test, forward=df_test['forward'][0])
    # df_angle = df_test.iloc[:, 33:49]
    print(f"We calculated that the {df_test['forward'][0]} foot is the forward foot.")
    return df_test


def rename_columns(df, forward="LEFT"):
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
    df_pose['forward'] = df_pose.apply(
        lambda x: front_leg(x['LEFT_HIP'][0:3], x['RIGHT_HIP'][0:3], x['RIGHT_SHOULDER'][0:3],
                            x['LEFT_FOOT_INDEX'][0:3], x['RIGHT_FOOT_INDEX'][0:3]), axis=1)

    df_pose_LEFT = df_pose[df_pose['forward'] == 'LEFT']
    df_pose_RIGHT = df_pose[df_pose['forward'] == 'RIGHT']
    df_pose_LEFT = rename_columns(df_pose_LEFT)
    df_pose_RIGHT = rename_columns(df_pose_RIGHT, forward="RIGHT")
    df_corrected = pd.concat([df_pose_LEFT, df_pose_RIGHT], axis=0)

    return df_corrected, df_corrected.columns[35:51]


def create_pose_df():
    df_world = load_pickle(PICKLEDPATH, "pose_world_landmark_all_df.pickle")
    df_world = df_world.replace(to_replace='None', value=np.nan).dropna()
    df_world = df_world.reset_index(drop=True)

    create_angle_features(df_world)

    df_w1 = df_world.loc[df_world['label'] == 1].copy()
    df_w2 = df_world.loc[df_world['label'] == 2].copy()
    df_dd = df_world.loc[df_world['label'] == 0].copy()

    return df_w1, df_w2, df_dd


def create_pose_np():
    np_world = load_pickle(PICKLEDPATH, "pose_world_landmark_numpy.pickle")
    return np_world


def get_angle_confidence_intervals(df_pose: pd.DataFrame, LANDMARKS: list, percent: float = .10) -> dict:
    df_pose = df_pose.loc[df_pose['quality'] == 1]
    num = df_pose.shape[0]
    num_to_take = int(num - num * (1 - percent))

    valid_angles = {}
    for idx, col in enumerate(LANDMARKS):
        low = df_pose[col].sort_values(ascending=True).reset_index(drop=True)[num_to_take]
        median = df_pose[col].median()
        high = df_pose[col].sort_values(ascending=True).reset_index(drop=True)[num - num_to_take]

        valid_angles[col] = [low, median, high]

    return valid_angles


def calc_2d_angles(vec_2d):
    return np.degrees(np.arctan2(vec_2d[1], vec_2d[0]))


def calc_xyz_angles(vec):
    xy_points = vec[0:2]
    yz_points = vec[1:3]

    xy_angle = calc_2d_angles(xy_points)
    yz_angle = calc_2d_angles(yz_points)

    return xy_angle, yz_angle


def calc_xyz_dist(vec):
    xy_points = vec[0:2]
    yz_points = vec[1:3]

    xy_dist = np.linalg.norm(xy_points, axis=0)
    yz_dist = np.linalg.norm(yz_points, axis=0)

    return xy_dist, yz_dist


def calc_xyz_points(xy_angle, xy_dist, yz_angle, yz_dist):
    x, y1 = calc_2d_points(xy_angle, xy_dist)
    y2, z = calc_2d_points(yz_angle, yz_dist)

    return x, y1, z


def calc_2d_points(angle, dist):
    a = dist * np.cos(np.radians(angle))
    b = dist * np.sin(np.radians(angle))
    return a, b


def get_annotated_img(indx):
    annotated_images = load_pickle(PICKLEDPATH, "annotated_images.pickle")
    annotated_images = [img for img in annotated_images if img is not None]
    return annotated_images[indx]


def normalize_on_right_hip(keypoints):
    RIGHT_HIP = 24

    keypoints = keypoints - np.tile(np.swapaxes(keypoints[:, RIGHT_HIP, :][np.newaxis, :], 0, 1),
                                    (1, keypoints.shape[1], 1))
    return keypoints


def select_correct_closest_image(np_test, df):
    good_idx = df[df['quality'] == 1].index
    np_world = create_pose_np()
    np_good = np_world[good_idx]

    np_good = normalize_on_right_hip(np_good)
    np_test = normalize_on_right_hip(np_test)

    small_indx = np.argmin(np.sum(np.linalg.norm((np_good - np_test), axis=2), axis=1))
    ground_truth = np_good[small_indx, :, :]
    ground_truth_indx = good_idx[small_indx]

    return ground_truth, ground_truth_indx


def compare_two_figures(length_fig, angle_fig, img1, img2, plot=True):
    plot_image_grid([img1, img2], 2)

    xy_dist, yz_dist = calc_xyz_dist(length_fig.T)
    xy_angle, yz_angle = calc_xyz_angles(angle_fig.T)

    px, py, pz = calc_xyz_points(xy_angle, xy_dist, yz_angle, yz_dist)
    lx, ly, lz = length_fig.T[0], length_fig.T[1], length_fig.T[2]
    angx, angy, angz = angle_fig.T[0], angle_fig.T[1], angle_fig.T[2]

    if plot:
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(px, py, pz)
        ax.scatter3D(lx, ly, lz)
        # ax.scatter3D(angx, angy, angz)
        ax.view_init(elev=-50, azim=270)
        # ax.plot(points[0],points[1],points[2],color = 'g')
        for i, j in BODY_POSE_CONNECTIONS:
            ax.plot([px[i], px[j]], [py[i], py[j]], [pz[i], pz[j]], color='b')
            ax.plot([lx[i], lx[j]], [ly[i], ly[j]], [lz[i], lz[j]], color='r')
            # ax.plot([angx[i], angx[j]], [angy[i], angy[j]], [angz[i], angz[j]], color='g')
        plt.show()