import numpy as np

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """
        self.training_data = training_data
        self.training_labels = training_labels
        pred_labels = self.predict( self.training_data )
        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        N = len( test_data )
        if self.task_kind == "classification":
            test_labels = self.kNN( test_data )
        else:
            test_labels = np.empty( ( N , 2 ) )

            for i in range( N ):
                distances = self.euclidean_dist( test_data[i] )
                idxs = self.find_k_nearest_neighbors( distances ) 
                n_distances = distances[idxs]  
                n_ys = self.training_labels[idxs]  
                if n_distances[0] == 0:
                    test_labels[i] = np.mean( n_ys )
                else:
                    weights = 1 / n_distances
                    test_labels[i] = np.sum( weights[:, np.newaxis]*n_ys, axis=0 ) / np.sum( weights )

        return test_labels


    def euclidean_dist(self, example):
        squared_diffs = np.square( example - self.training_data )
        result = np.sqrt( np.sum( squared_diffs , axis=1 ) )
        return result   

    def find_k_nearest_neighbors(self, distances):
        sorted_array = np.argsort( distances )
        indices = sorted_array[:self.k]
        return indices 

    def predict_label(self, neighbor_labels):
        occurrences_array=np.bincount(neighbor_labels)
        return np.argmax(occurrences_array)  

    def kNN_one_example(self, unlabeled_example):
        distances = self.euclidean_dist( unlabeled_example )
        nn_indices = self.find_k_nearest_neighbors( distances )
        neighbor_labels = self.training_labels[nn_indices]
        best_label = self.predict_label( neighbor_labels )
        return best_label

    def kNN(self, unlabeled):
        return np.apply_along_axis( func1d= self.kNN_one_example , axis=1 , arr=unlabeled )