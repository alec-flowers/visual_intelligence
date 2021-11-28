import numpy as np
import torch


def evaluate_model(model, dataset, reshape_inputs=False):
    inputs = dataset.coordinates
    targets = dataset.labels
    if reshape_inputs:
        inputs = torch.from_numpy(dataset.coordinates)
        inputs = inputs.view(inputs.size(0), -1).float()
        targets = torch.from_numpy(dataset.labels)

    model.eval()
    with torch.no_grad():
        # Predict training data
        predictions = model(inputs)
        predicted_class = np.argmax(predictions, axis=1)
        return targets, predicted_class
