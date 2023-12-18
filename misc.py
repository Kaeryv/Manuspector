import torch
import numpy as np

def data_augment(x, betashift=0.05, slopeshift=0.05, multishift=0.05):
    # Shift of baseline
    # calculate arrays
    beta = np.random.random(size=(x.shape[0], 1)) * 2 * betashift - betashift
    slope = np.random.random(size=(x.shape[0], 1)) * 2 * slopeshift - slopeshift + 1
    # Calculate relative position
    axis = np.array(range(x.shape[1])) / float(x.shape[1])
    # Calculate offset to be added
    offset = slope * (axis) + beta - axis - slope / 2. + 0.5

    # Multiplicative
    multi = np.random.random(size=(x.shape[0], 1)) * 2 * multishift - multishift + 1

    x = multi * x + offset
    return(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def torchify(X, dtype=torch.float, cc=None):
    if cc is not None:
        return torch.from_numpy(np.expand_dims(X, cc)).type(dtype)
    else:
        return torch.from_numpy(X).type(dtype)
