import pickle
import os
import pathlib

REPO_ROOT = pathlib.Path(__file__).absolute().parents[0].resolve()
assert (REPO_ROOT.exists())
TRAINPATH = (REPO_ROOT / "data" / "train").absolute().resolve()
assert (TRAINPATH.exists())
TESTPATH = (REPO_ROOT / "data" / "test").absolute().resolve()
assert (TESTPATH.exists())


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
