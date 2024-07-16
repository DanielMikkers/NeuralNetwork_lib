# Neural Network

## ToDo's
| TODO Layers | | | |
| --------- | --------- | --------- | --------- |
| Code Dense Layer | X | Comment Dense Layer | | 
| Code RNN Layer | X | Comment RNN Layer | |
| Code LSTM Layer | | Comment LSTM Layer | | 
| Code GRU Layer | | Comment GRU Layer | |
| Code Conv1D Layer | | Comment Conv1D Layer | |
| Code Conv2D Layer | | Comment Conv2D Layer | |
| Code ConvLSTM 1D Layer | | Comment ConvLSTM 1D Layer | | 
| Code ConvLSTM 2D Layer | | Comment ConvLSTM 2D Layer | | 
| Code ConvLSTM 3D Layer | | Comment ConvLSTM 3D Layer | | 
| Code AveragePooling1D Layer | | Comment AveragePooling1D Layer | |
| Code AveragePooling2D Layer | | Comment AveragePooling2D Layer | |
| Code AveragePooling3D Layer | | Comment AveragePooling3D Layer | |
| Code MaxPooling1D Layer | | Comment MaxPooling1D Layer | |
| Code MaxPooling2D Layer | | Comment MaxPooling2D Layer | |
| Code MaxPooling3D Layer | | Comment MaxPooling3D Layer | |
| Code Dropout Layer | | Comment Dropout Layer | |
| Code Flatten Layer | | Comment Flatten Layer | | 
| Code Embedding Layer | | Comment Embedding Layer | |
| Code Attention Layer | | Comment Attention Layer | | 
| Code MutliHeadAttention Layer | | Comment MultiHeadAttention Layer | | 
| Code BatchNormalization Layer | | Comment BatchNormalization Layer | |

| TODO NN | | | |
| --------- | --------- | --------- | --------- |
| Code KNN | | Comment KNN | |
| Code Q-learning | | Comment Q-Learning | |
| Code LLM for trading | | Comment LLM for trading | |
| Code Linear Regression model | | Comment Linear Regression model | |
| Code fitting model | | Comment fitting model | |

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



## Long-Short-Term-Memory (LSTM) Layer