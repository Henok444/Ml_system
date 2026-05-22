import numpy as np

class LinearRegressionGD:
    def __init__(self):


        pass


    def fit(self,X , y ,lr , epochs):
        self.lr = lr
        self.epochs = epochs 

        n_samples , n_features = X.shape
        w = np.zeros(n_features) 
        b = 0   
        tol = 1e-6
        for i in range(self.epochs):
            y_pred = np.dot(X , w) + b   # although w is 1D series  numpy convert it to column matrix automatically 
            error = y - y_pred 
            jw = (-2/n_samples)* np.dot(X.T, error)
            jb = (-2/n_samples) * np.sum(error)
            w = w - self.lr * jw 
            b = b - self.lr  * jb
            if np.all(np.abs(jw) < tol) and abs(jb) < tol:
                break

        self.weight = w
        self.bias = b

    def predict(self , X_new):
        y_new = self.bias +  np.dot(X_new , self.weight)
        return y_new


