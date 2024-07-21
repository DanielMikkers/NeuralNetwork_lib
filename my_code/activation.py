from numpy import ndarray as Tensor
import numpy as np

class ActivationFunction:
    def softmax(self, x: Tensor, **kwargs) -> Tensor:
        a = np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(x - a)
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def softmax_grad(self, x: Tensor, **kwargs) -> Tensor:
        p = self.softmax(x)
        return p * (1 - p)

    def sigmoid(self, x: Tensor, **kwargs) -> Tensor:
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_grad(self, x: Tensor, **kwargs) -> Tensor:
        p = self.sigmoid(x)
        return p * (1 - p)
    
    def tanh(self, x: Tensor, **kwargs) -> Tensor:
        return np.tanh(x)
    
    def tanh_grad(self, x: Tensor, **kwargs) -> Tensor:
        return 1 - np.power(self.tanh(x),2)

    def relu(self, x: Tensor, **kwargs) -> Tensor:
        return np.where(x>=0, x, 0)
    
    def relu_grad(self, x: Tensor, **kwargs) -> Tensor:
        return np.where(x>=0, 1, 0)
    
    def leakyRelu(self, x: Tensor, alpha: float = 0.2):
        return np.where(x>=0, x, alpha*x)
    
    def leakyRelu_grad(self, x: Tensor, alpha: int = 0.2):
        return np.where(x>=0, 1, alpha*1)
    
    def linear(self, x: Tensor, **kwargs) -> Tensor:
        return x
    
    def linear_grad(self, x: Tensor, **kwargs) -> Tensor:
        return np.ones(np.shape(x))
    
    def elu(self, x: Tensor, alpha: float = 0.1) -> Tensor:
        return np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))
    
    def elu_grad(self, x: Tensor, alpha: float = 0.1) -> Tensor:
        return np.where(x >= 0.0, 1, self.alpha * np.exp(x))
    
    def silu(self, x: Tensor, **kwargs) -> Tensor:
        return x*self.sigmoid(x)
    
    def silu_grad(self, x: Tensor, **kwargs) -> Tensor: 
        return self.sigmoid(x) + x*self.sigmoid_grad(x)
    
    def softplus(self, x: Tensor, **kwargs) -> Tensor:
        return np.log(np.exp(x) + 1)
    
    def softplus_grad(self, x: Tensor, **kwargs) -> Tensor:
        return 1 / (1 + np.exp(-x))
    
    def softsign(self, x: Tensor, **kwargs) -> Tensor:
        return x / (np.absolute(x) + 1)
    
    def softsign_grad(self, x: Tensor, **kwargs) -> Tensor:
        eps = 1e-15
        return 1 / (np.absolute(x) + 1) - np.power(x, 2) / (np.absolute(x+eps)*np.power((np.absolute(x) + 1),2))