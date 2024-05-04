from tensor import Tensor
from typing import Dict, Union, Tuple
from activation import activefunc
import numpy as np

class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError
    
    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError
    
class Linear(Layer):

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]
    
    def backward(self, grad: Tensor) -> Tensor:
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad

        return grad @ self.params["w"]

class Dense(Layer):
    def __init__(self, units: int, activation: str = 'relu', input_shape: Union[Tuple[int],None] = None) -> None:
        super().__init__()

        self.units = units
        self.activation = activation
        self.input_shape = input_shape
        self.need_input_shape = input_shape is None
        self.f, self.f_prime = self.func(activation)

        self.params["w"] = None
        self.params["b"] = None


    def func(self, activation: str) -> Tuple[function]:
        self.active_func = {
            'tanh': (activefunc.tanh, activefunc.tanh_prime),
            'sigmoid': (activefunc.sigmoid, activefunc.sigmoid_prime),
            'relu': (activefunc.ReLU, activefunc.ReLU_prime),
            'prelu': (activefunc.pReLU, activefunc.pReLu_prime),
            'softmax': (activefunc.softmax, activefunc.softmax_prime)
        }

        if activation in self.active_func:
            self.activation_function = self.active_func[activation]
            return self.activation_function
        else:
            raise ValueError("Unknown activation function called: {}".format(activation))

    def linear(self, inputs: Tensor) -> Tensor:
        return inputs @ self.params["w"] + self.params["b"]
    
    def forward(self, inputs: Tensor) -> Tensor:
        if self.params["w"] is None:
            input_shape = input.shape[1:]
            self.params["w"] = np.random.randn(np.prod(input_shape), self.units)
            self.params["b"] = np.random.randn(self.units)
        self.inputs = inputs
        z = self.linear(inputs)
        return self.f(z)
    
    def backward(self, grad: Tensor) -> Tensor:
        grad = grad * self.activation(self.inputs, derivative=True)
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"] 


#class Activation(Layer):
#    def __init__(self, f: F, f_prime: F) -> None:
#        super().__init__()
#        self.f = f
#        self.f_prime = f_prime
#    
#    def forward(self, inputs: Tensor) -> Tensor:
#        self.inputs = inputs
#        return self.f(inputs)
#    
#    def backward(self, grad: Tensor) -> Tensor:
#        return self.f_prime(self.inputs) * grad
#
#class Dense(Layer):
#    def backward(self, grad: Tensor) -> Tensor:
#        grad = grad * self.activation(self.inputs, derivative=True)
#        self.grads["b"] = np.sum(grad, axis=0)
#        self.grads["w"] = self.inputs.T @ grad
#        return grad @ self.params["w"]
#    
