# Importing the dependencies

import numpy as np

class SVM_classifier():
    
    # initialing hyperparameters
    def __init__(self, learning_rate, no_of_itearions, lambda_parameter):
        
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_itearions
        self.lambda_parameter = lambda_parameter
    
    # fitting the dataset to SVM classifier
    def fit(self, X, Y):
        
        # m --> number of data points --> no. of rows,      n --> number of input features --> no. of columns
        self.m , self.n = X.shape
        
        #initialing weight and bias
        self. w = np.zeros(self.n)
        self. b = 0
        self. X = X
        self. Y = Y
        
        # implementing Gradient Descent Algorithm for Optimization
        
        for i in range(self.no_of_iterations):
            self.update_weights()
      
     # Function for updating the weight and bias value
     
    def update_weights(self):
        
        y_label = np.where(self.Y <= 0, -1, 1)   # Y_label --> Yi
        
        # gradients (dw, db)
        for index, x_i in enumerate(self.X):
            
            condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1
            
            if (condition == True):
                
                dw = 2 * self.lambda_parameter * self.w
                db = 0
            
            else:
                
                dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index])
                db = np.dot(y_label[index]) 
            
            # Updating weights and bias 
            self.w = self.w - self.learning_rate * dw
        
            self.b = self.b - self.learning_rate * db
                
    # predict the label for a given input value
    def predict(self, X):
        
        output = np.dot(X, self.w) - self.b  # Hyperplane equation -->  y = mx - b
        
        predicted_labels = np.sign(output)   # rounding the number if value is 1, or 2 , 0r 3 then it'll be 1
                                             # if value is -1 , -10 , or -0.87 then it'll be -1
                                             
        y_hat = np.where(predicted_labels <= -1, 0, 1)
        
        return y_hat