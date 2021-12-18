from sklearn import tree
import numpy as np
from data.data_loading import create_angle_features
import pickle
import os

from pose.pose_utils import load_pickle, PICKLEDPATH, LANDMARKS_ANGLES_DICT, DECISION_TREE_PATH


def train_decision_tree(pose: str):
    """
    Train a decision tree to classify a pose into good or bad pose. We can then use the decision tree nodes to give corrections.
    :param pose: Pose to train the decision tree for
    :return: Saves decision tree to saved_model/
    """
    if pose == "DD":
        pose_int = 0
    elif pose == "W1":
        pose_int = 1
    elif pose == "W2":
        pose_int = 2
    # prepare the dataset
    df_world = load_pickle(PICKLEDPATH, "pose_world_landmark_all_df.pickle")
    df_world = df_world.replace(to_replace='None', value=np.nan).dropna()
    create_angle_features(df_world)
    # subset to pose
    df_world_pose = df_world.loc[df_world['label'] == pose_int]
    # prepare for training
    X_train = df_world_pose[LANDMARKS_ANGLES_DICT]
    y_train = df_world_pose['quality']
    # train decision tree
    clf = tree.DecisionTreeClassifier(max_depth=3, random_state=12)
    clf = clf.fit(X_train, y_train)
    # save decision tree
    with open(os.path.join(DECISION_TREE_PATH, f'{pose}_decision_tree.pickle'), 'wb') as f:
        pickle.dump(clf, f)


def decision_tree_correct(df_pose, pose: str):
    """
    Classifies the passed pose as good or bad and and gives back corrections determined by decision tree.
    :param pose: Pose to correct
    """
    if pose == "DD":
        pose_int = 0
    elif pose == "W1":
        pose_int = 1
    elif pose == "W2":
        pose_int = 2
    # prepare the dataset
    df_world = load_pickle(PICKLEDPATH, "pose_world_landmark_all_df.pickle")
    df_world = df_world.replace(to_replace='None', value=np.nan).dropna()
    create_angle_features(df_world)
    # subset to pose
    df_world_pose = df_world.loc[df_world['label'] == pose_int]
    # prepare for training
    X_train = df_world_pose[LANDMARKS_ANGLES_DICT]
    y_train = df_world_pose['quality']
    # train decision tree
    clf = tree.DecisionTreeClassifier(max_depth=3, random_state=12)
    clf = clf.fit(X_train, y_train)
    # do correction
    X_df = df_pose[LANDMARKS_ANGLES_DICT]
    X = X_df.to_numpy()
    print_decision_path(clf, X, X_df.columns)


def print_decision_path(clf, X, feature_names):
    """
    Taken from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#decision-path.
    Gets the decision path taken by decision tree to give corrections and prints out the nodes that lead to
    classification.
    """
    # init
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_indicator = clf.decision_path(X)
    leaf_id = clf.apply(X)
    sample_id = 0
    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[
                 node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
                 ]
    print("Rules used to predict sample {id}:\n".format(id=sample_id))
    for node_id in node_index:
        # continue to the next node if it is a leaf node
        if leaf_id[sample_id] == node_id:
            continue
        # check if value of the split feature for sample 0 is below threshold
        if X[sample_id, feature[node_id]] <= threshold[node_id]:
            threshold_sign = "<="
        else:
            threshold_sign = ">"
        print(
            "decision node {node} : (X_test[{sample}, {feature}] = {value}) "
            "{inequality} {threshold})".format(
                node=node_id,
                sample=sample_id,
                feature=feature_names[feature[node_id]],
                value=X[sample_id, feature[node_id]],
                inequality=threshold_sign,
                threshold=threshold[node_id],
            )
        )


if __name__ == '__main__':
    train_decision_tree("W2")
