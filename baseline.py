import numpy as np
import pandas as pd

from data.data import create_angle_features
from utils import PICKLEDPATH, load_pickle


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
