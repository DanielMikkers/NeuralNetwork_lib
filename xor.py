import numpy as np
from train import train
from nn import Sequential
from layers import Dense

inputs = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

targets = np.array([
    [1,0],
    [0,1],
    [0,1],
    [1,0]
])

net = Sequential([])
net.add_layer(Dense(units =32, activation = 'relu', input_shape=2))
net.add_layer(Dense(units=2, activation='tanh'))

train(net, inputs, targets)

for x,y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)