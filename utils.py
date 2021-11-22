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

POSEDATAFRAME_LIST = ["pose_landmark_all_df.pickle", "pose_landmark_vis_df.pickle",
                      "pose_landmark_numpy.pickle", "pose_world_landmark_all_df.pickle",
                      "pose_world_landmark_vis_df.pickle", "pose_world_landmark_numpy.pickle",
                      "labels_drop_na.pickle", "annotated_images.pickle"]

"""
0: Downward Dog
1: Warrior I
2: Warrior II
"""
CLASS_MAPPINGS_NAMES = {
    0: "DD",
    1: "W1",
    2: "W2",
    3: "DD",
    4: "W1",
    5: "W2",
}
CLASS_MAPPINGS_IDX = {
    0: 0,
    1: 1,
    2: 2,
    3: 0,
    4: 1,
    5: 2,
}
"""
0: bad
1: good
"""
POSE_QUALITY_MAPPINGS = {
    0: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 1,
    5: 1,
}

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

# https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/pose_connections.py

POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                      (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                      (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                      (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                      (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                      (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                      (29, 31), (30, 32), (27, 31), (28, 32)]

BODY_POSE_CONNECTIONS = [(11, 12), (11, 13),
                          (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                          (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                          (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                          (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                          (29, 31), (30, 32), (27, 31), (28, 32)]

LANDMARKS_ANGLES_DICT = {
    'LEFT_ELBOW_ANGLE': [11, 13, 15],
    'RIGHT_ELBOW_ANGLE': [12, 14, 16],
    'LEFT_ARMPIT_ANGLE': [13, 11, 23],
    'RIGHT_ARMPIT_ANGLE': [14, 12, 24],
    'LEFT_CHEST_ANGLE': [13, 11, 12],
    'RIGHT_CHEST_ANGLE': [14, 12, 11],
    'LEFT_WRIST_ANGLE': [19, 15, 13],
    'RIGHT_WRIST_ANGLE': [20, 16, 14],
    'LEFT_KNEE_ANGLE': [23, 25, 27],
    'RIGHT_KNEE_ANGLE': [24, 26, 28],
    'LEFT_HIPFLEXOR_ANGLE': [11, 23, 25],
    'RIGHT_HIPFLEXOR_ANGLE': [12, 24, 26],
    'LEFT_ADDUCTOR_ANGLE': [24, 23, 25],
    'RIGHT_ADDUCTOR_ANGLE': [23, 24, 26],
    'LEFT_ANKLE_ANGLE': [25, 27, 31],
    'RIGHT_ANKLE_ANGLE': [26, 28, 32],
}
LANDMARKS_ANGLES_DICT = {
    key: [LANDMARK_DICT[idx] for idx in value] for key, value in LANDMARKS_ANGLES_DICT.items()
}


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
        try:
            lms.append(np.array([lm['x'], lm['y'], lm['z']]))
        except TypeError:
            lms.append(np.array([lm[0], lm[1], lm[2]]))
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
