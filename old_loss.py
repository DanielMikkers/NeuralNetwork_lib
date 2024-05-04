import numpy as np

class Losses:

    def BinaryCrossEntropy(self, y_true, y_pred):

        return 1

    def MSE(self, y_true, y_pred):
        """
        Classes mean square error
        """

        loss = np.mean( np.square((y_true - y_pred)) )
    
        return loss
    
    def MAE(self, y_true, y_pred):
        """
        Classes mean absolute error
        """

        loss = np.mean(np.absolute(y_true - y_pred))

        return loss
    
    def MSLE(self, y_true, y_pred):
        """
        Classes mean squared logarithmic error
        """

        loss = np.mean( np.square(np.log( y_true + 1 ) - np.log(y_pred + 1) ))

        return loss
    
    def CosineSimilarities(self, y_true, y_pred, axis=-1):
        """
        Classes
        """
        
        dot_prod =  np.sum(y_true * y_pred, axis=axis)
        norm_true = np.sqrt(np.sum(y_true * y_true, axis=axis))
        norm_pred = np.sqrt(np.sum(y_pred * y_pred, axis=axis))

        loss = dot_prod / (norm_true * norm_pred)

        return loss
    
    def LogCosh(self, y_true, y_pred, axis=None):
        """
        Classes
        """

        error = y_pred - y_true
        loss = np.mean( np.log( (np.exp(error) + np.exp(-error))/2 ) , axis=axis)

        return loss

    def mean_squared_error(self, y_true, y_pred, axis):
        """
        Function
        """

        loss = np.mean( np.square((y_true - y_pred) ), axis=axis)

        return loss

    def mean_absolute_error(self, y_true, y_pred, axis=-1):
        """
        Function mean absolute error
        """

        loss = np.mean(np.absolute(y_true - y_pred), axis=axis)

        return loss
    
    def mean_square_logarithmic_error(self, y_true, y_pred, axis=-1):
        """
        Function mean squared logarithmic error
        """

        loss = np.mean( np.square(np.log( y_true + 1 ) - np.log(y_pred + 1) ), axis=axis)

        return loss
    
    def cosine_similarities(self, y_true, y_pred, axis=-1):
        """
        Function
        """
        
        dot_prod =  np.sum(y_true * y_pred, axis=axis)
        norm_true = np.sqrt(np.sum(y_true * y_true, axis=axis))
        norm_pred = np.sqrt(np.sum(y_pred * y_pred, axis=axis))

        loss = dot_prod / (norm_true * norm_pred)

        return loss
    
    def log_cosh(self, y_true, y_pred, axis=None):
        """
        Classes
        """

        error = y_pred - y_true
        loss = np.mean( np.log( (np.exp(error) + np.exp(-error))/2 ) , axis=axis)

        return loss

loss = Losses()