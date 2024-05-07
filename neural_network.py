from loss import MeanSquareError, CrossEntropy
from numpy import ndarray
import numpy as np
import tqdm
from utils import batch_iterator, shuffle_data
from terminaltables import AsciiTable
from optimizer import SGD
import time

loss_functions = {
    'mean_square_error': (MeanSquareError.loss, MeanSquareError.accuracy, MeanSquareError.gradient),
    'cross_entropy': (CrossEntropy.loss, CrossEntropy.accuracy, CrossEntropy.gradient)
}

optimizer_dict = {
    'SGD': SGD
}

class Sequential:
    def __init__(self, optimizer: str = 'SGD', loss: str = 'mean_square_error', validation = None, shuffle = False) -> None:
        self.optimizer = optimizer_dict[optimizer]
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.acc = {"training": [], "validation": []}
        self.loss_func = loss_functions[loss][0]
        self.accuracy_func = loss_functions[loss][1]
        self.gradient_func = loss_functions[loss][2]
        self.shuffle = shuffle
        
        self.val_set = None
        if validation is not None:
            x, y = validation
            self.val_set = {"X": x, "y": y}
    
    def trainable_set(self, trainable):
        for layer in self.layers: 
            layer.tranable = trainable
        
    def add_layer(self, layer): 
        if self.layers:
            layer.set_input_shape(shape=self.layers[-1].output_shape())
        
        if hasattr(layer, 'initialize'):
            layer.initialize(optimizer=self.optimizer)
        
        self.layers.append(layer)
    
    def test_batch(self, x, y_true):
        y_pred = self.forward_pass(x)
        loss = np.mean(self.loss_func(y_true, y_pred))
        accuracy = self.accuracy_func(y_true, y_pred)

        return loss, accuracy
    
    def train_batch(self, x, y_true):
        y_pred = self._forward(x)
        loss = self.loss_func(y_true, y_pred)
        accuracy = self.loss_func(y_true, y_pred)

        loss_grad = self.gradient_func(y_true, y_pred)

        self._backward(loss_grad=loss_grad)

        return loss, accuracy
    
    def fit(self, x, y_true, epochs, batch_size):
        for _ in tqdm(range(100), desc="Training", unit="epoch", ncols=75, colour='#37B6BD'):
            time.sleep(0.1)

            batch_error = []
            batch_acc = []
            for x_batch, y_batch in batch_iterator(x, y_true, batch_size=batch_size):
                if self.shuffle:
                    x_batch, y_batch = shuffle_data(x_batch, y_batch, seed=None)
                loss, acc = self.train_batch(x_batch, y_batch)
                batch_error.append(loss)
                batch_acc.append(acc)

            self.acc["training"].append(np.mean(batch_acc))
            self.errors["training"].append(np.mean(batch_error))

            if self.val_set is not None:
                val_loss, val_acc = self.test_batch(self.val_set["x"], self.val_set["y"])
                self.errors["validation"].append(val_loss)
                self.acc["validation"].append(val_acc)
        
        return self.error["training"], self.errors["validation"], self.acc["training"], self.acc["validation"]

    def _forward(self, x, train=True):
        layer_output = x
        for layer in self.layers:
            layer_output = layer.forward(layer_output, train)
        return layer_output
    
    def _backward(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)

    def summary(self, name: str = "Model"):
        print(AsciiTable([[name]].table))
        
        table_data = [["Layer (type)", "Output Shape", "Params"]]
        tot_params = 0

        for layer in self.layers:
            layer_name = layer.layer_name()
            params = layer.parameters()
            output_shape = layer.output_shape()
            table_data.append([str(layer_name), str(params), str(output_shape)])
            tot_params += params
        
        print(AsciiTable(table_data).table)
        print("Total paramaters: %d \n" % tot_params)
    
    def predict(self, x):
        return self._forward(x, traning=False)
    
    def save_weights(self, file_name: str):
        return 0
    
    def import_weights(self, file_name: str):
        return 0