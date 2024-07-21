# Neural Network

## ToDo's
| TODO Layers | | | | | |
| --------- | --------- | --------- | --------- | --------- | --------- |
| Code Dense Layer | X | Comment Dense Layer | X | Check correctness with TF | |
| Code RNN Layer | X | Comment RNN Layer | | Check correctness with TF | |
| Code LSTM Layer | | Comment LSTM Layer | | Check correctness with TF | |
| Code GRU Layer | | Comment GRU Layer | | Check correctness with TF | |
| Code Conv1D Layer | | Comment Conv1D Layer | | Check correctness with TF | |
| Code Conv2D Layer | | Comment Conv2D Layer | | Check correctness with TF | |
| Code ConvLSTM 1D Layer | | Comment ConvLSTM 1D Layer | | Check correctness with TF | |
| Code ConvLSTM 2D Layer | | Comment ConvLSTM 2D Layer | | Check correctness with TF | |
| Code ConvLSTM 3D Layer | | Comment ConvLSTM 3D Layer | | Check correctness with TF | |
| Code AveragePooling1D Layer | | Comment AveragePooling1D Layer | | Check correctness with TF | |
| Code AveragePooling2D Layer | | Comment AveragePooling2D Layer | | Check correctness with TF | |
| Code AveragePooling3D Layer | | Comment AveragePooling3D Layer | | Check correctness with TF | |
| Code MaxPooling1D Layer | | Comment MaxPooling1D Layer | | Check correctness with TF | |
| Code MaxPooling2D Layer | | Comment MaxPooling2D Layer | | Check correctness with TF | |
| Code MaxPooling3D Layer | | Comment MaxPooling3D Layer | | Check correctness with TF | |
| Code Dropout Layer | | Comment Dropout Layer | | Check correctness with TF | |
| Code Flatten Layer | | Comment Flatten Layer | | Check correctness with TF | |
| Code Embedding Layer | | Comment Embedding Layer | | Check correctness with TF | |
| Code Attention Layer | | Comment Attention Layer | | Check correctness with TF | |
| Code MutliHeadAttention Layer | | Comment MultiHeadAttention Layer | | Check correctness with TF | |
| Code BatchNormalization Layer | | Comment BatchNormalization Layer | | Check correctness with TF | |

| TODO NN | | | |
| --------- | --------- | --------- | --------- | --------- | --------- |
| Code KNN | | Comment KNN | | Check correctness with TF | |
| Code Physics informed NN | | Comment Physics informed NN | | Check correctness with TF | |
| Code Q-learning | | Comment Q-Learning | | Check correctness with TF | |
| Code LLM | | Comment LLM | | Check correctness with TF | |
| Code Linear Regression model | | Comment Linear Regression model | | Check correctness with TF | |
| Code fitting model | | Comment fitting model | | Check correctness with TF | |

NOTE: look into econometrics or finance informed machine learning / neural networks.

## Introduction
Let me define a few things first:
- Let $\boldsymbol{h}(\mathbf{x}) \in \mathbb{R}^n$ be a function. I will call this function the activation function;
- Let $W,V,U$ be some matrices - which are not necessarily square matrices - which will be called the weights;
- Let $\mathbf{b},\mathbf{c}$ be some vectors which will be called the biases;
- Let $L$ be the loss function;
- Let $\boldsymbol{o}$ be the output;
- In the code I will call $\boldsymbol{h}'(\mathbf{z}) \nabla_{\boldsymbol{o}} L$: ```grad_z```;
- In the code I will call $\nabla_{\boldsymbol{o}} L$: ```grad```;

## Dense Layer
A dense layer is just a simple layer for a NN. Let $\mathbf{x}$ be the input of the dense layer. Then the output of the dense layer $\mathbf{o}$ is given by (when a bias is present):

$$\boldsymbol{o} = \boldsymbol{h}(\mathbf{x} W + \mathbf{b}).$$

This was the forward pass. Now we need to do the backward pass and update the weights. The gradient of the weight $W$ is then

$$\nabla_W L = \frac{\partial L}{\partial W} = \frac{\partial \boldsymbol{o}}{\partial W} \frac{\partial L}{\partial \boldsymbol{o}} = \mathbf{x}^T \boldsymbol{h}'(\mathbf{z}) \nabla_{\boldsymbol{o}} L$$

$$\nabla_\mathbf{b} L = \frac{\partial L}{\partial \mathbf{b}} = \frac{\partial \boldsymbol{o}}{\partial \mathbf{b}} \frac{\partial L}{\partial \boldsymbol{o}} = \boldsymbol{h}'(\mathbf{z}) \nabla_{\boldsymbol{o}} L$$

Then the weights $W$ and the bias $\mathbf{b}$ (if bias is turned on) are updated by the updating rule of the specific optimizer. The function of the backward pass returns $\left(\boldsymbol{h}'(\mathbf{z}) \nabla_{\boldsymbol{o}} L \right) \cdot W^T$, since this is the derivative w.r.t. the input $\mathbf{x}$. 

## Recurrent Neural Network (RNN) Layer

An RNN can learn on timeseries like data. The architechture of an RNN is given in the figure below.

<p align="center">
<img src="https://github.com/DanielMikkers/NeuralNetwork_lib/blob/main/RNN.png" width="50%" height="50%">
</p>

Let me define the following

$$\boldsymbol{a}^{(t)} = \mathbf{b} + W \boldsymbol{h}^{(t-1)} + U \mathbf{x} $$
$$\boldsymbol{h}^{(t)} = f(\boldsymbol{a}^{(t)})$$
$$\boldsymbol{o}^{(t)} = \mathbf{c} + V \boldsymbol{h}^{(t)}$$

Then computing the gradients:

$$\nabla_{\mathbf{b}} L = \sum_t \boldsymbol{h}'^{(t)} \nabla_{\boldsymbol{h}^{(t)}} L$$
$$\nabla_{\mathbf{c}} L = \sum_t \nabla_{\boldsymbol{o}^{(t)}} L$$
$$\nabla_{U} L = \sum_t \boldsymbol{h}'^{(t)} \left(\nabla_{\boldsymbol{h}^{(t)}} L \right) \mathbf{x}^T$$
$$\nabla_{V} L = \sum_t \boldsymbol{h}'^{(t)} \left(\nabla_{\boldsymbol{h}^{(t)}} L \right) \boldsymbol{h}^{(t-1),T}$$
$$\nabla_{W} L = \sum_t (\nabla_{\boldsymbol{o}^{(t)}} L) \boldsymbol{h}^{(t),T}$$

The gradient w.r.t. $\boldsymbol{o}^{(t)}$ is the 'regular' gradient of $L$. The gradient w.r.t. $\boldsymbol{h}^{(t)}$ is given by

$$\nabla_{\boldsymbol{h}^{(t)}} L = \boldsymbol{h}^{(t)} (\nabla_{\boldsymbol{o}^{(t)}} L) \cdot V $$

The gradient for the next layer is 

## Long-Short-Term-Memory (LSTM) Layer
