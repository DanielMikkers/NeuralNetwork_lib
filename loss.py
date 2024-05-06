from numpy import ndarray as Tensor
import numpy as np
from utils import get_accuracy

class Loss(object):
    def loss(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        raise NotImplementedError
    
    def gradient(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        raise NotImplementedError
    
    def accuracy(self, y_true: Tensor, y_pred: Tensor) -> float:
        return 0
    
class MeanSquareError(Loss):
    def __init__(self) -> None:
        pass

    def loss(self, y_true: Tensor, y_pred: Tensor) -> float: 
        return np.mean(0.5 * np.power((y_true - y_pred),2))
    
    def accuracy(self, y_true: Tensor, y_pred: Tensor) -> float:
        accuracy = get_accuracy(y_true, y_pred)
        return accuracy
    
    def gradient(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return -(y_true - y_pred)

class CrossEntropy(Loss):
    def __init__(self) -> None:
        pass

    def loss(self, y_true: Tensor, y_pred: Tensor) -> float:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -y_true * np.log(y_pred) - (1-y_true) * np.log(1 - y_pred)
    
    def accuracy(self, y_true: Tensor, y_pred: Tensor) -> float:
        accuracy = get_accuracy(y_true, y_pred)
        return accuracy
    
    def gradient(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return - y_true / y_pred + (1-y_true) / (1-y_pred)