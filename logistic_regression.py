"""
Simple and multinominal Logistic Regression
Author: Tanay Tunçer

"""

import numpy as np

class LogisticRegression():

    def __init__(self, epochs = 1000, feature_scaling = False, reg = False, learning_rate = 0.01, C = 0.01, binary_class = True):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.w = None
        self.b = None
        self.reg = reg
        self.C = C
        self.binary_class = binary_class
        self.feature_scaling = feature_scaling

    
    def linear_transformation(self, X, w, b):
       """
            Calculate linear function: x · w + b
            Args:
                X (m, n):
                w (m, 1):
                b (n):
            Return:
                linear function
       """ 
       return np.dot(X, self.w) + self.b

    def activation_function(self, z):
        """
            Calculate sigmoid or softmax function. 
            The function differenciate beetween the sigmoid function for binary output and softmax function for multiclass output.
            Args:
                z (ndarray): Logit
            Return 
                Compute a value between 0 and 1. 
        """
        if self.binary_class == True:
            return 1 / (1 + np.exp(-z))
        else:
            return np.exp(z) / np.sum(np.exp(z), axis = 0)

    def cost_function(self, y, h):
        """
            Calculate binary cross entropy loss.
            Args:
                y (ndarray): Actual output
                h (ndarray): Predicted output 
            Return:
                Compute the total cost of execution (scalar)
        """
        return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h)) 
           
    def gradient_descent(self, X, y, h):
        """
            Calculate gradients and execute gradient descent to compute weights and the bias term.
            Args:
                X (m,n): Input values
                y (m,): Output labels
                h (m,): Predicted output labels 
            Return:
                w (ndarray), b (n): Compute new weights and bias term.
        """ 
        m, n = X.shape
        
        if self.reg == False:
            dw = (1/m) * np.dot(X.T, (h - y))                     
            db = (1/m) * np.sum(h - y)
        else:
            dw = ((1/m) * np.dot(X.T, (h - y))) + np.dot(self.learning_rate, self.w)                   
            db = (1/m) * np.sum(h - y)

        self.w = self.w - (dw * self.learning_rate)
        self.b = self.b - (db * self.learning_rate)

        return self.w, self.b
        
    def fit(self, X, y):
        """
            Training the logisitic regression model. 
            Args:
                X (m, n): Input values
                y (m,): Output labels 
        """
                 
        m, n = X.shape 
        self.w = np.random.rand(n)
        self.b = 0.
        l_wb = np.zeros([self.epochs, 2])

        X = ((X.T - np.mean(X, axis = 1)) / np.std(X, axis = 1)).T if self.feature_scaling == True else X
        
        for i in range(self.epochs):
            h = self.activation_function(self.linear_transformation(X, self.w, self.b))
            self.w, self.b = self.gradient_descent(X, y, h)
            
            if self.reg == False:
                l_wb[i] = [i, self.cost_function(y, h)] 
            else: 
                l_wb[i] = [i, self.cost_function(y, h) + ((self.C / (2*m)) * np.sum(np.square(self.w)**2))]
                                                                           
            if i % (self.epochs / 10) == 0:
                print(f'The cost after epoch number {i} is: {np.round(l_wb[i][1], 6)}')
            
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

        X = ((X.T - np.mean(X, axis = 1)) / np.std(X, axis = 1)).T if self.feature_scaling == True else X

        predictions = self.activation_function(self.linear_transformation(X, self.w, self.b))

        if self.binary_class == True:
            h = np.where(predictions > 0.5, 1, 0)
        else:
            h = np.argmax(predictions, axis = 1) 

        return h