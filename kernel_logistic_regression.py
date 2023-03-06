"""
Kernel Logistic Regression
Author: Tanay Tunçer

"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class KernelLogisticRegression():

    def __init__(self, learning_rate = 0.01, gamma = 0.7, epochs = 100, C = 0.01):
        self.learning_rate = learning_rate
        self.C = C
        self.epochs = epochs
        self.alpha = None
        self.b = None
        self.gamma = gamma
        self.K = None

    def linear_transformation(self, K, w, b):
       """
            Calculate linear function: x · w + b
            Args:
                X (m, n):
                w (m, 1):
                b (n):
            Return:
                linear function
       """ 
       return np.dot(K, self.alpha) + self.b

    def activation_function(self, z):
        """
            Calculate sigmoid or softmax function. 
            The function differenciate beetween the sigmoid function for binary output and softmax function for multiclass output.
            Args:
                z (ndarray): Logit
            Return 
                Compute a value between 0 and 1. 
        """
        return 1 / (1 + np.exp(-z))
    
    @staticmethod 
    def radial_basis_function_kernel(x1, x2, gamma):
        return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

    def cost_function(self, y, h):
        """
            Calculate binary cross entropy loss.
            Args:
                y (ndarray): Actual output
                h (ndarray): Predicted output 
            Return:
                Compute the total cost of execution (scalar).
        """
        return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))    

    def gradient_descent(self, K, y, h):
        """
            Calculate gradients and execute gradient descent to compute weights and the bias term.
            Args:
                X (m,n): Input values
                y (m,): Output labels
                h (m,): Predicted output labels 
            Return:
                alpha (ndarray), b (n): Compute new weights and bias term.
        """ 
        m, _ = K.shape

        da = ((1/m) * np.dot(K.T, (h - y))) + (np.dot(self.learning_rate, self.alpha))               
        db = (1/m) * np.sum(h - y)
        
        self.alpha = self.alpha - (da * self.learning_rate)
        self.b = self.b - (db * self.learning_rate)
                
        return self.alpha, self.b
        
    def fit(self, X, y):
        """
            Training the logisitic regression model. 
            Args:
                X (m, n): Input values
                y (m,): Output labels 
        """      
        m, _ = X.shape 
        
        self.K = np.zeros((m,m))
        self.alpha = np.zeros(m)
        self.b = 0
        
        l_wb = np.zeros([self.epochs, 2]) 
        
        for i in range(m):
            for j in range(m):
                self.K[i,j] = self.radial_basis_function_kernel(X[i],X[j], self.gamma)
        
 
        for e in range(self.epochs):
            h = self.activation_function(self.linear_transformation(self.K, self.alpha, self.b))
            self.alpha, self.b = self.gradient_descent(self.K, y, h)
            
            l_wb[e] = [e, (self.cost_function(y, h) + ((self.C / (2*m)) * np.sum(np.square(self.alpha))))]

            if e % (self.epochs / 10) == 0:
                print(f'The cost after epoch number {e} is: {np.round(l_wb[e][1], 6)}')
        
        h = np.where(h > 0.5, 1, 0)
        
        return l_wb, h

    
    def predict(self, X):
        """
            Predict new data.
            Args:
                X: Input values, features
            Return:
                y: Output predicted y label for input data. 
        """ 
    
        predictions = np.zeros(X.shape[0])
        alpha = self.alpha[:X.shape[0]]
        K = np.zeros((X.shape[0], X.shape[0]))
        
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                K[j,i] = self.radial_basis_function_kernel(X[j], X[i], self.gamma)
                predictions = self.activation_function(np.dot(K, alpha) + self.b)

        h = np.where(predictions > 0.5, 1, 0)
        
        return h



