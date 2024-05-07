from numpy import ndarray as Tensor
import numpy as np
from neural_network import Sequential
    
class SGD():
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.1) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dw = None

    def update(self, w: Tensor, grad_w: Tensor) -> Tensor:
        if self.dw is None:
            self.dw = np.zeros(np.shape(w))

        self.dw = self.momentum * self.dw + (1 - self.momentum) * grad_w
        return w - self.learning_rate * self.dw

