import pickle
import os
import pathlib

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
                  'LEFT_FOOD_INDEX',
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
