import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500, task_kind="classification"):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        onehotlabels = label_to_onehot( training_labels )
        self.weights = np.random.normal( 0 , 0.1 , ( training_data.shape[1] , onehotlabels.shape[1] ) )

        for iteration in range( self.max_iters ):
            grad = self.gradient( training_data , onehotlabels , self.weights )
            self.weights = self.weights - self.lr*grad
            
        pred_labels = self.predict( training_data )
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        pred_labels = np.argmax( self.f_softmax( test_data , self.weights ) , axis = 1 )
        return pred_labels

    def f_softmax(self, data, W):
        """
        Softmax function for multi-class logistic regression.
        
        Args:
            data (array): Input data of shape (N, D)
            W (array): Weights of shape (D, C) where C is the number of classes
        Returns:
            array of shape (N, C): Probability array where each value is in the
                range [0, 1] and each row sums to 1.
                The row i corresponds to the prediction of the ith data sample, and 
                the column j to the jth class. So element [i, j] is P(y_i=k | x_i, W)
        """
        num = np.exp( data @ W )
        return num / np.sum( num , axis = 1 , keepdims = True )

    def gradient(self, data, labels, W):
        """
        Compute the gradient of the entropy for multi-class logistic regression.
        
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            W (array): Weights of shape (D, C)
        Returns:
            grad (np.array): Gradients of shape (D, C)
        """
        return data.T @ ( self.f_softmax( data , W ) - labels )

