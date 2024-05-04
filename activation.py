from tensor import Tensor
import numpy as np

class ActivationFunc:
    def tanh(self, x: Tensor, **kwargs) -> Tensor:
        return np.tanh(x)

    def tanh_prime(self, x: Tensor, **kwargs) -> Tensor:
        y = np.tanh(x)
        return 1-y**2

    def sigmoid(self, x: Tensor, **kwargs) -> Tensor:
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x: Tensor, **kwargs) -> Tensor:
        y = np.exp(-x)
        return y/(1+y)**2

    def ReLU(self, x: Tensor, **kwargs) -> Tensor:
        return np.maximum(0, x)

    def ReLU_prime(self, x: Tensor, **kwargs) -> Tensor:
        y = x
        y[y>0] = 1
        y[y<= 0] = 0
        return y

    def pReLU(self, x: Tensor, a: float = 0.01) -> Tensor:
        min_x = np.minimum(0, x)
        max_x = np.maximum(0, x)

        return a * min_x + max_x

    def pReLU_prime(self, x: Tensor, a: float = 0.01) -> Tensor:
        y = x
        y[y>0] = 1
        y[y<=0] = a

        return y
    
    def softmax(self, x: Tensor, **kwargs) -> Tensor:
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return y
    
    def softmax_prime(self, x: Tensor, **kwargs) -> Tensor:
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return y
    
    def linear(self, x: Tensor, **kwargs) -> Tensor:
        return x
    
    def swish(self, x: Tensor, **kwargs) -> Tensor:
        return x * self.sigmoid(x)
    

    
activefunc = ActivationFunc()