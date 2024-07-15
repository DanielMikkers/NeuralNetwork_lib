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
    def __init__(self, units, activation= 'linear', input_shape = None, bias = False) -> None:
        self.layer_input = None
        self.input_shape = input_shape
        self.units = units
        self.trainable = True
        self.w = None
        self.b = None
        self.bias = bias
        self.activation_name = activation

    def initialize(self, optimizer):
        limit = 1 / np.sqrt(self.input_shape[0])
        self.w = np.random.uniform(-limit,limit, (self.input_shape[0], self.units))
        self.b = np.zeros((1,self.units))

        self.w_opt = copy.copy(optimizer)
        self.b_opt = copy.copy(optimizer)

        self.activation_func, self.activation_grad = activation_function[self.activation_name]

    def parameters(self):
        if self.bias is False:
            return np.prod(np.shape(self.w))
        else:
            return np.prod(np.shape(self.w)) + np.prod(np.shape(self.b))
    
    def forward(self, inputs, training = True):
        self.layer_input = inputs
        if self.bias is False:
            output = inputs.dot(self.w)
        else:
            output = inputs.dot(self.w) + self.b
        return self.activation_func(output)
    
    def backward(self, grad):
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

class RNN(Layer):
    def __init__(self, units, activation = 'linear', input_shape = None, bptt_trunc = 5, bias = False):
        self.layer_input = None
        self.input_shape = input_shape
        self.units = units
        self.trainable = True
        self.bptt_trunc = bptt_trunc
        self.W = None
        self.V = None
        self.U = None
        self.b = None
        self.c = None
        self.bias = bias
        self.activation_name = activation
    
    def initialize(self, optimizer):
        timesteps, input_dim = self.input_shape

        limit_dim = 1 / np.sqrt(self.input_dim)
        limit_unit = 1 / np.sqrt(self.units)
        self.U = np.random.uniform(-limit_dim, limit_dim, (self.units, input_dim))
        self.W = np.random.uniform(-limit_unit, limit_unit, (self.units, self.units))
        self.V = np.random.uniform(-limit_unit, limit_unit, (input_dim, self.units))
        self.b = np.zeros((1,self.units))
        self.c = np.zeros((1,self.input_shape))

        self.U_opt = copy.copy(optimizer)
        self.W_opt = copy.copy(optimizer)
        self.V_opt = copy.copy(optimizer)
        self.b_opt = copy.copy(optimizer)
        self.c_opt = copy.copy(optimizer)

        self.activation_func, self.activation_grad = activation_function[self.activation_name]

    def parameters(self):
        if self.bias is False:
            return np.prod(self.W.shape) + np.prod(self.U.shape) + np.prod(self.V.shape)
        else:
            return np.prod(self.W.shape) + np.prod(self.U.shape) + np.prod(self.V.shape) + np.prod(self.b.shape) + np.prod(self.c.shape)
    
    def forward(self, inputs, training = True):
        self.layer_input = inputs
        batch_size, timesteps, input_dim = np.shape(inputs)

        self.state_input = np.zeros((batch_size, timesteps, self.units))
        self.states = np.zeros((batch_size, timesteps+1, self.units))
        self.outputs = np.zeros((batch_size, timesteps, input_dim))

        self.states[:, -1] = np.zeros((batch_size, self.units))

        if self.bias is False:
            self.b = np.zeros_like(self.b)
            self.c = np.zeros_like(self.c)

        for t in range(timesteps):
            self.state_input[:, t] = self.b + self.layer_input[:, t].dot(self.U.T) + self.states[:, t-1].dot(self.W.T)
            self.states[:, t] = self.activation_func(self.state_input[:, t])
            self.outputs[:, t] = self.c + self.states[:, t].dot(self.V.T)

        return self.outputs

    def backward(self, grad):
        _, timesteps, _ = np.shape(grad)

        grad_U = np.zeros_like(self.U)
        grad_V = np.zeros_like(self.V)
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)
        grad_c = np.zeros_like(self.c)

        fut_grad = np.zeros_like(grad)

        for t in reversed(range(timesteps)):
            grad_V += grad[:, t].T.dot(self.states)

    def output_shape(self):
        return self.input_shape


class LSTM(Layer):
    def one():
        return 1

class Attention(Layer):
    def one():
        return 1

class AveragePooling1D(Layer):
    def one():
        return 1
    
class AveragePooling2D(Layer):
    def one():
        return 1
    
class AveragePooling3D(Layer):
    def one():
        return 1
    
class BatchNormalization(Layer):
    def one():
        return 1

class Conv1D(Layer):
    def one():
        return 1

class Conv2D(Layer):
    def one():
        return 1

class ConvLSTM1D(Layer):
    def one():
        return 1
    
class ConvLSTM2D(Layer):
    def one():
        return 1
    
class ConvLSTM3D(Layer):
    def one():
        return 1
    
class Dropout(Layer):
    def one():
        return 1

class Embedding(Layer):
    def one():
        return 1

class Flatten(Layer):
    def one():
        return 1

class GRU(Layer):
    def one():
        return 1

class MaxPooling1D(Layer):
    def one():
        return 1

class MaxPooling2D(Layer):
    def one():
        return 1

class MaxPooling3D(Layer):
    def one():
        return 1

class MultiHeadAttention(Layer):
    def one():
        return 1

class SimpleRNN(Layer):
    def one():
        return 1

