from tensor import Tensor
from layers import Layer
from typing import Sequence, Iterator, Tuple
import numpy as np

class Sequential:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers
    
    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad
    
    def add_layer(self, layer: Layer) -> None:
        if self.layers:
            input_shape = self.layers[-1].units
            layer.input_shape = input_shape
        self.layers.append(layer)