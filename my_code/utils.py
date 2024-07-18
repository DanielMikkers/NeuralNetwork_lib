from numpy import ndarray as Tensor
import numpy as np

def get_accuracy(y_true: Tensor, y_pred: Tensor):
    accuracy = np.sum(y_true == y_pred) / np.size(y_true)
    return  accuracy

def batch_iterator(x, y_true = None, batch_size=41):
    n_samples = np.shape(x)[0]
    for i in np.arange(0,n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y_true is not None:
            yield x[begin:end], y_true[begin:end]
        else:
            yield x[begin:end]

def shuffle_data(x, y, seed=None):
    if seed:
        np.random.seed(seed)
    idx = np.arange(np.shape(x)[0])
    np.random.shuffle(idx)
    return x[idx], y[idx]

def split_test_train(x, y, test_size=0.20, shuffle=True, seed=None):
    if shuffle:
        x, y = shuffle_data(x, y, seed)
    
    split_idx = int(test_size * len(y))
    x_train, x_test = x[split_idx:], x[:split_idx]
    y_train, y_test = y[split_idx:], y[:split_idx]

    return x_train, x_test, y_train, y_test