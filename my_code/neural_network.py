from .loss import MeanSquareError, CrossEntropy
from numpy import ndarray
import numpy as np
from tqdm import tqdm
from .utils import batch_iterator, shuffle_data
from terminaltables import AsciiTable
from .optimizer import SGD
import time

#dictionary of loss functions, to assign later to variables
loss_functions = {
    'mean_square_error': (MeanSquareError().loss, MeanSquareError().accuracy, MeanSquareError().gradient),
    'cross_entropy': (CrossEntropy().loss, CrossEntropy().accuracy, CrossEntropy().gradient)
}

#dictionary of optimizers
optimizer_dict = {
    'SGD': SGD()
}


class Sequential:
    def __init__(self, optimizer: str = 'SGD', loss: str = 'mean_square_error', validation = None, shuffle = False, name: str = "Sequential") -> None:
        self.optimizer = optimizer_dict[optimizer]
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.acc = {"training": [], "validation": []}
        self.loss_func = loss_functions[loss][0]
        self.accuracy_func = loss_functions[loss][1]
        self.gradient_func = loss_functions[loss][2]
        self.shuffle = shuffle
        self.mod_name = name
        
        self.val_set = None
        if validation is not None:
            x, y = validation
            self.val_set = {"X": x, "y": y}
    
    def trainable_set(self, trainable):
        for layer in self.layers: 
            layer.trainable = trainable
        
    def add_layer(self, layer): 
        #function to add layers one by one to the model
        if self.layers:
            layer.set_input_shape(shape=self.layers[-1].output_shape())
        
        if hasattr(layer, 'initialize'):
            layer.initialize(optimizer=self.optimizer)
        
        self.layers.append(layer)
    
    def test_batch(self, x, y_true):
        #test batch for accuracy and loss
        y_pred = self._forward(x)
        loss = np.mean(self.loss_func(y_true, y_pred))
        accuracy = self.accuracy_func(y_true, y_pred)

        return loss, accuracy
    
    def train_batch(self, x, y_true):
        #train batch to update weights, and obtain loss and accuracy
        y_pred = self._forward(x)
        
        loss = np.mean(self.loss_func(y_true, y_pred))
        accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)

        loss_grad_new = self.gradient_func(y_true, y_pred)

        self._backward(loss_grad=loss_grad_new)

        return loss, accuracy
    
    def fit(self, x, y_true, epochs, batch_size):
        i=1
        
        #fit the model to the input data
        for _ in tqdm(range(epochs)):#, desc="Training", unit="epoch", ncols=75, colour='#37B6BD'):
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
            
            print(f"Epoch {i}: training error = {np.mean(batch_error)}, training accuracy = {np.mean(batch_acc)} \n")
            i+=1

            if self.val_set is not None:
                val_loss, val_acc = self.test_batch(self.val_set["x"], self.val_set["y"])
                self.errors["validation"].append(val_loss)
                self.acc["validation"].append(val_acc)
        
        return self.errors["training"], self.errors["validation"], self.acc["training"], self.acc["validation"]

    
    def _forward(self, x):
        #call the forward function of the layers
        layer_output = x
        for layer in self.layers:
            layer_output = layer.forward(layer_output)

        return layer_output
    
    def _backward(self, loss_grad):
        #call the backward function of the layers
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)
            

    def summary(self, name: str = "Model: "):
        #print a summary of the model giving the layer types, number of parameters and the shapes
        Mod_Name = name + self.mod_name
        print(AsciiTable([[Mod_Name]]).table)
        
        table_data = [["Layer (type)", "Output Shape", "Params"]]
        tot_params = 0

        for layer in self.layers:
            layer_name = layer.layer_name()
            params = layer.parameters()
            output_shape = layer.output_shape()
            table_data.append([str(layer_name), str(output_shape), str(params)])
            tot_params += params
        
        print(AsciiTable(table_data).table)
        print("Total paramaters: %d \n" % tot_params)
    
    def predict(self, x):
        #predict the output given input x
        return self._forward(x)
    
    def save_weights(self, file_name: str):
        #TODO: function to save the weights to a file
        return 0
    
    def import_weights(self, file_name: str):
        #TODO: function to import the weights to a fike
        return 0