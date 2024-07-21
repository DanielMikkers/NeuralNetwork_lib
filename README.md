# Neural Network

## ToDo's
| Layer Type | Code | Comment | Check correctness with TF |
| --------- | --------- | --------- | --------- |
| Dense Layer | X | X | X |
| RNN Layer | X | | |
| LSTM Layer | | | |
| GRU Layer | | |
| Conv1D Layer | | |
| Conv2D Layer | | |
| ConvLSTM 1D Layer | | |
| ConvLSTM 2D Layer | | |
| ConvLSTM 3D Layer | | |
| AveragePooling1D Layer | | |
| AveragePooling2D Layer | | |
| AveragePooling3D Layer | | |
| MaxPooling1D Layer | | |
| MaxPooling2D Layer | | |
| MaxPooling3D Layer | | |
| Dropout Layer | | |
| Flatten Layer | | |
| Embedding Layer | | |
| Attention Layer | | |
| MutliHeadAttention Layer | | |
| BatchNormalization Layer | | |

| Model Type | Code | Comment | Check correctness with TF| 
| --------- | --------- | --------- | --------- |
| KNN | | |
| Physics informed NN | | |
| Q-learning | | |
| LLM | | |
| Linear Regression model | | |
| fitting model | | |

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

### Maths
A dense layer is just a simple layer for a NN. Let $\mathbf{x}$ be the input of the dense layer. Then the output of the dense layer $\mathbf{o}$ is given by (when a bias is present):

$$\boldsymbol{o} = \boldsymbol{h}( W \mathbf{x} + \mathbf{b}),$$

where $W$ are some weights and $\mathbf{b}$ is a bias. This is the forward pass of a dense layer. 

The weights $W$ are initialized by taking a random matrix with dimensions ($N$, units), where $N$ is the first dimension of the input shape, which is the ???????? for the first layer and the number of units for next layers. The random matrix is drawn from the uniform distribution $\text{UNIF}(-\frac{1}{\sqrt{\text{N}}}, \frac{1}{\sqrt{\text{N}}})$.

Now we need to do the backward pass and update the weights. The gradient of the weight $W$ is then

$$\nabla_W L = \frac{\partial L}{\partial W} = \frac{\partial \boldsymbol{o}}{\partial W} \frac{\partial L}{\partial \boldsymbol{o}} = \mathbf{x}^T \boldsymbol{h}'(\mathbf{z}) \nabla_{\boldsymbol{o}} L$$

$$\nabla_\mathbf{b} L = \frac{\partial L}{\partial \mathbf{b}} = \frac{\partial \boldsymbol{o}}{\partial \mathbf{b}} \frac{\partial L}{\partial \boldsymbol{o}} = \boldsymbol{h}'(\mathbf{z}) \nabla_{\boldsymbol{o}} L$$

The function of the backward pass returns $\left(\boldsymbol{h}'(\mathbf{z}) \nabla_{\boldsymbol{o}} L \right) \cdot W^T$, since this is the derivative w.r.t. the input $\mathbf{x}$. 

Then the weights $W$ and the bias $\mathbf{b}$ (if bias is turned on) are updated by the updating rule of the specific optimizer.

### Comparing with TensorFlow
The code has been checked by comparing the loss and accuracy with a TensorFlow/Keras model. The architecture was as follows:

1. Flattening layer with input shape (32,32,3)
2. Dense (hidden) layer with ReLU as activation function and 16 units
3. Dense (output) Layer with softmax as activation function and 10 units, one for each class.

The models were trained on the cifar10 dataset from TensorFlow. The models ran for 5 epochs. The loss for each epoch were quite similar. The accuracy of the TensorFlow model was slightly more consistent for each test round and performed generally sligtly better. The accuracy of the TensorFlow model was, after 5 epochs, between $0.08$ and $0.11$. My model performed less consistent and showed accuracies ranging from $0.01$ to $0.10$.

**This may be due to the initialization of the matrices.**

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
Check correctness with TF