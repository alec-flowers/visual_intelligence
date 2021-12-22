from typing import Tuple, Union, Any

import numpy as np
import torch
from numpy import ndarray
from torch import nn, Tensor

from data.data_loading import CoordinatesDataset


def evaluate_model(model: nn.Module, dataset: CoordinatesDataset,
                   reshape_inputs: bool = False, good_bad: bool = False) \
        -> Tuple[Union[Tensor, Any], Tensor]:
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
    :param good_bad: whether we are in the pose quality classification scenario or not
    :type good_bad:
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
        if good_bad:
            predicted_class = torch.round(torch.sigmoid(predictions))
        else:
            predicted_class = torch.argmax(predictions, dim=1)
        return targets, predicted_class
