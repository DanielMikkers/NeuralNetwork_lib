from tensor import Tensor
import numpy as np

class Loss:
    def loss(self, y_true: Tensor, y_pred: Tensor) -> float:
        raise NotImplementedError
    
    def grad(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        raise NotImplementedError
    
class MSE(Loss):
    def loss(self, y_true: Tensor, y_pred: Tensor) -> float:
        return np.sum((y_true - y_pred)**2)
    
    def grad(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return 2 * (y_true - y_pred)