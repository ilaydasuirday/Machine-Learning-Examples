import numpy as np


class Perceptron(object): 
 """Perception Classifier 
 Parameters
 ----------
 era: float (Learning Rate -between 0.0 and 1.0
 n_iter: int (Passes over the training data- set)
 
 Attributes
 ----------
 w_ : 1d array (weights after sitting)
 errors_ : list (number of misclassification every epach)"""

 def __init__(self, eta = 0.01, n_iter = 10):
    """learning rate (n) is 0.01 (Q=n(yi-yi')xi)"""
    self.eta = eta
    self.n_iter = n_iter

 def fit(self, X, y):
    """Fit training data 
    Parameters
    ----------
    X : {array-like}, shape = [n_samples, n_features]
    Training vectors, where 
    n_samples is the number of samples and Training Machine Learning Algorithms for Classification
    n_features is the number of features.
    
    y : array-like, shape = [n_samples]
    Target values.
    
    returns 
    -------
    self: object"""

    self.w_ = np.zeros(1 + X.shape[1])
    """np.zeros(shape, data type) : sıfırlarla dolu bir array oluşturmak için kullanılır"""
    self.errors_ = []
    for _ in range (self.n_iter):
        erros = 0
        for xi, target in zip(X,y):
            update = self.eta * (target-self.predict(xi))
            self.w_[1:] += update * xi
            self.w_[0] += update
            erros += int(update != 0.0)
            self.errors_.append(erros)
    return self

 def net_input(self, X):
    """calculate net input"""
    return np.dot(X, self.w_[1:]) + self.w_[0]
    """The np.dot function that is used calculates the vector dot product z=w.x"""

 def predict(self, X):
    """return class label after unit step"""
    return np.where(self.net_input(X) >= 0.0, 1, -1)







