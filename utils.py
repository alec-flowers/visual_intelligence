import pickle
import os
import pathlib
import numpy as np

REPO_ROOT = pathlib.Path(__file__).absolute().parents[0].resolve()
assert (REPO_ROOT.exists())
DATAPATH = (REPO_ROOT / "data").absolute().resolve()
assert (DATAPATH.exists())
TRAINPATH = (DATAPATH / "train").absolute().resolve()
assert (TRAINPATH.exists())
TESTPATH = (DATAPATH / "test").absolute().resolve()
assert (TESTPATH.exists())
PICKLEDPATH = (DATAPATH / "pickled_data").absolute().resolve()
assert (PICKLEDPATH.exists())
MODEL_PATH = (REPO_ROOT / "saved_model/mlp").absolute().resolve()
assert (MODEL_PATH.exists())
PLOT_PATH = (REPO_ROOT / "plots").absolute().resolve()
assert (PLOT_PATH.exists())

POSEDATAFRAME_LIST = ["pose_landmark_all_df.pickle", "pose_landmark_vis_df.pickle",
                      "pose_landmark_numpy.pickle", "pose_world_landmark_all_df.pickle",
                      "pose_world_landmark_vis_df.pickle", "pose_world_landmark_numpy.pickle",
                      "labels_drop_na.pickle", "annotated_images.pickle"]

LANDMARK_NAMES = ['NOSE',
                  'LEFT_EYE_INNER',
                  'LEFT_EYE',
                  'LEFT_EYE_OUTER',
                  'RIGHT_EYE_INNER',
                  'RIGHT_EYE',
                  'RIGHT_EYE_OUTER',
                  'LEFT_EAR',
                  'RIGHT_EAR',
                  'MOUTH_LEFT',
                  'MOUTH_RIGHT',
                  'LEFT_SHOULDER',
                  'RIGHT_SHOULDER',
                  'LEFT_ELBOW',
                  'RIGHT_ELBOW',
                  'LEFT_WRIST',
                  'RIGHT_WRIST',
                  'LEFT_PINKY',
                  'RIGHT_PINKY',
                  'LEFT_INDEX',
                  'RIGHT_INDEX',
                  'LEFT_THUMB',
                  'RIGHT_THUMB',
                  'LEFT_HIP',
                  'RIGHT_HIP',
                  'LEFT_KNEE',
                  'RIGHT_KNEE',
                  'LEFT_ANKLE',
                  'RIGHT_ANKLE',
                  'LEFT_HEEL',
                  'RIGHT_HEEL',
                  'LEFT_FOOT_INDEX',
                  'RIGHT_FOOT_INDEX']
num = []
for i in range(33):
    num.append(i)
LANDMARK_DICT = dict(zip(num, LANDMARK_NAMES))


def save_pickle(data, path, file):
    """Save a file as .pickle"""
    filename = os.path.join(path, file)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path, file):
    """Load pickle file"""
    file_path = os.path.join(path, file)
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_dataframes_to_pickle(path, dataframes, filenames):
    assert len(dataframes) == len(filenames)
    for i in range(len(dataframes)):
        save_pickle(dataframes[i], path, filenames[i])


def calc_angle(lm_1, lm_2, lm_3, ref=np.array([0, 1, 0])):
    """
    Calculates angle between three landmarks in reference to the y-norm vector.
    :param lm_1: Landmark 1
    :param lm_2: Landmark 2
    :param lm_3: Landmark 3
    :param ref: Reference vector to determine angle from 0° to 360°
    :return: 0° <= angle < 360°
    """
    # landmark prep
    lms = [lm_1, lm_2, lm_3]
    for lm in lms.copy():
        lms.pop(0)
        lms.append(np.array([lm['x'], lm['y'], lm['z']]))
    lm_1 = lms[0]
    lm_2 = lms[1]
    lm_3 = lms[2]
    # get the vector with reference to lm_2
    lm_2_lm_1_vector = lm_1 - lm_2
    lm_2_lm_3_vector = lm_3 - lm_2
    # calc angle using https://ch.mathworks.com/matlabcentral/answers/501449-angle-betwen-two-3d-vectors-in-the-range-0-360-degree
    cross = np.cross(lm_2_lm_3_vector, lm_2_lm_1_vector)
    sign = np.sign(np.dot(cross, ref))
    if sign == 0:
        raise ValueError('Reference Vector, v1 and v2 are in the same plane!')
    c = sign * np.linalg.norm(cross)
    angle = np.degrees(np.arctan2(c, np.dot(lm_2_lm_3_vector, lm_2_lm_1_vector)))
    # get angle between 0 - 360
    corrected_angle = (angle + 360) % 360
    return corrected_angle
