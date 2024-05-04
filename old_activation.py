import numpy as np

class ActivationFunction:    
    
    def ELU(self, x, alpha):
        min_x = np.minimum(0, x)
        max_x = np.maximum(0, x)

        p = alpha * (np.exp(min_x) - 1) + max_x

        return p
    
    def SELU(self, x, alpha, scaled):
        min_x = np.minimum(0, x)
        max_x = np.maximum(0, x)

        p = scaled * alpha * (np.exp(min_x) - 1) + scaled * max_x

        return p
    
    def ReLU_Shift(self, x, shift):
        p = np.maximum(shift, x)

        return p

    def softplus(self, x):
        p = np.log(np.exp(x) + 1)
        return p
    
    def softsign(self, x):
        p = x / (np.absolute(x) + 1)
        return p
    
    def Swish(self, x):
        p = x * self.sigmoid(x)

        return p