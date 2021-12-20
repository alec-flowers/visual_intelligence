from typing import Tuple

import numpy as np
import torch
from torch import nn

from data.data_loading import CoordinatesDataset


def evaluate_model(model: nn.Module, dataset: CoordinatesDataset, reshape_inputs: bool = False) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Make predictions with the trained model.
    :param model: the MLP or CNN model
    :type model: nn.Module
    :param dataset: the data to make predictions on
    :type dataset: CoordinatesDataset
    :param reshape_inputs: for the MLP we need to reshape the input format
    :type reshape_inputs: bool
    :return: the targets and the predictions
    :rtype: Tensor
    """
    inputs = dataset.coordinates
    targets = dataset.labels
    if reshape_inputs:
        inputs = torch.from_numpy(dataset.coordinates)
        inputs = inputs.view(inputs.size(0), -1).float()
        targets = torch.from_numpy(dataset.labels)

    model.eval()
    with torch.no_grad():
        predictions = model(inputs)
        predicted_class = np.argmax(predictions, axis=1)
        return targets, predicted_class
