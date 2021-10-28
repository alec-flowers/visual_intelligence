import pickle
import os

TRAINPATH = "./data/train"
TESTPATH = "./data/test"


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
