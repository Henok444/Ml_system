import numpy as np 


class LinearRegressionSGD:


    def __init__(self):
        
        pass
    def fit(self ,  X , y, epochs , alpha):
        
        n , col = X.shape 

        random_no_list = np.random.permutation(n)
        self.slope = np.zeros(col)
        self.intercept = 0
        self.w = []
        self.b= []
        
        for epoch in range(epochs):

            for i in random_no_list:
                self.y_pred = np.dot(X[i,:] , self.slope.T) + self.intercept 
                error = y.iloc[i] - self.y_pred

                gradient_slope = (-2)*(error * X[i,:])
                gradient_intercept = (-2) * (error)

                self.slope -= alpha * gradient_slope
                self.intercept -= alpha * gradient_intercept
            self.w.append(self.slope)
            self.b.append(self.intercept)
        return self.slope , self.intercept
    def predict(self , X_new):
        y_new = np.dot(X_new , self.slope.T) + self.intercept
        return y_new