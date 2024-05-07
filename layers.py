from numpy import ndarray as Tensor
import numpy as np
from activation import ActivationFunction
from typing import Tuple, Union
import copy

class Layer(object):
    def set_input_shape(self, shape: Tuple):
        self.input_shape = shape
    
    def layer_name(self):
        return self.__class__.__name__
    
    def parameters(self):
        return 0

    def forward(self, x: Tensor, training: bool):
        raise NotImplementedError
    
    def backward(self, x: Tensor, accum_grad):
        raise NotImplementedError
    
    def output_shape(self):
        raise NotImplementedError
    
activation_function = {
    'softmax': (ActivationFunction.softmax, ActivationFunction.softmax_grad),
    'sigmoid': (ActivationFunction.sigmoid, ActivationFunction.sigmoid_grad),
    'tanh': (ActivationFunction.tanh, ActivationFunction.tanh_grad),
    'relu': (ActivationFunction.relu, ActivationFunction.relu_grad),
    'leaky': (ActivationFunction.leakyRelu, ActivationFunction.leakyRelu_grad),
    'linear': (ActivationFunction.linear, ActivationFunction.linear_grad),
    'elu': (ActivationFunction.elu, ActivationFunction.elu_grad),
    'silu': (ActivationFunction.silu, ActivationFunction.silu_grad),
    'softplus': (ActivationFunction.softplus, ActivationFunction.softplus_grad),
    'softsign': (ActivationFunction.softsign, ActivationFunction.softsign_grad)
}

class Dense(Layer):
    def __init__(self, units: int, activation: str = 'linear', input_shape: Union[Tuple[int],None] = None) -> None:
        self.layer_input = None
        self.input_shape = input_shape
        self.units = units
        self.trainable = True
        self.w = None
        self.b = None
        self.activation_name = activation

    def initialize(self, optimizer) -> None:
        limit = 1/ np.sqrt(self.input_shape[0])
        self.w = np.random.uniform(-limit,limit, (self.input_shape[0], self.units))
        self.b = np.zeros((1,self.units))

        self.w_opt = copy.copy(optimizer)
        self.b_opt = copy.copy(optimizer)

        self.activation_func, self.activation_grad = activation_function[self.activation_name]

    def parameters(self):
        return np.prod(np.shape(self.w)) + np.prod(np.shape(self.b))
    
    def forward(self, inputs: Tensor, training: bool = True) -> Tensor:
        self.layer_input = inputs
        output = inputs.dot(self.w) + self.b
        return self.activation_func(output)
    
    def backward(self, grad: Tensor) -> Tensor:
        w = self.w

        if self.trainable:
            grad_w = self.layer_input.T.dot(grad)
            grad_b = np.sum(grad)
            
            self.w = self.w_opt.update(self.w, grad_w)
            self.b = self.b_opt.update(self.b, grad_b)
        
        grad = grad.dot(w.T)
        return grad * self.activation_grad(self.layer_input)
    
    def output_shape(self):
        return (self.units,)
    
    def final_weights(self):
        return self.w, self.b
    
class Activation(Layer):
    def __init__(self, name: str = 'linear') -> None:
        self.activation_name = name
        self.activation_func, self.activation_grad = activation_function[name]
        self.trainable = True

    def forward(self, x: Tensor, training: bool) -> Tensor:
        self.layer_input = x
        return self.activation_func(x)
    
    def backward(self, grad: Tensor) -> Tensor:
        return grad * self.activation_grad(self.layer_input)
    
    def output_shape(self):
        return self.input_shape