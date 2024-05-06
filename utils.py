from numpy import ndarray as Tensor
import numpy as np

def get_accuracy(y_true: Tensor, y_pred: Tensor):
    accuracy = np.sum(y_true == y_pred) / np.size(y_true)
    return  accuracy