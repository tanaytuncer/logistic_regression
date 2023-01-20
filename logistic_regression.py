"""
Simple and multinomial Logistic Regression
Author: Tanay Tunçer

"""

import numpy as np

class LogisticRegression():

    def __init__(self, learning_rate = 0.01, epochs = 1000, feature_scaling = False, reg = False, lambda_ = 0.7, binary_class = True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None
        self.reg = reg
        self.lambda_ = lambda_
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

    #@staticmethod
    def activation_function(self, z):
        """
            Calculate sigmoid or softmax function. 
            The function differenciate beetween the sigmoid function for binary output and softmax function for multiclass outout.
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
                cost_value (scalar): Compute the total cost of execution.
        """
        return  -1 * np.mean(y * (np.log(h) - (1 - y) * np.log(1 - h)))

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
            dw = ((1/m) * np.dot(X.T, (h - y))) + np.dot(self.lambda_, self.w)                   
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
        self.w = np.random.rand(n) #np.zeros(n)
        self.b = 0
        l_wb = np.zeros([self.epochs, 2])

        X = ((X.T - np.mean(X, axis = 1)) / np.std(X, axis = 1)).T if self.feature_scaling == True else X
        

        for i in range(self.epochs):
            h = self.activation_function(self.linear_transformation(X, self.w, self.b))
            self.w, self.b = self.gradient_descent(X, y, h)

            if self.reg == False:
                l_wb[i] = [i, self.cost_function(y, h)] 
            else: 
                l_wb[i] = [i, (self.cost_function(y, h) + ((self.lambda_ / (2*m)) * np.sum(self.w ** 2)))]
                            
            if i % (self.epochs / 10) == 0:
                print(f'The cost after epoch number {i} is: {np.round(l_wb[i][1], 3)}')


        return l_wb

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
            h = [1 if prediction > 0.5 else 0 for prediction in predictions]
        else:
            h = np.argmax(predictions, axis = 1)

        return h
