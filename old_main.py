import numpy as np

class RNN:

    def __init__(self):
        self.hidden = hidden
        self.seq_length = seq_length
        self.x_shape = np.shape(x)

        self.Wx = np.random.uniform(-1/np.sqrt(self.hidden), np.sqrt(1/self.hidden))
        self.Wy = np.random.uniform(-1/np.sqrt(self.hidden), np.sqrt(1/self.hidden))
        self.V = np.random.uniform(-1/np.sqrt(self.hidden), np.sqrt(1/self.hidden))
        self.Wb = np.random.uniform(-1/np.sqrt(self.hidden), np.sqrt(1/self.hidden))
        self.Wc = np.random.uniform(-1/np.sqrt(self.hidden), np.sqrt(1/self.hidden))