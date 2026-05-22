import numpy as np 
import matplotlib.pyplot as plt 


class LinearRegressionNE:


    def __init__(self):
        

        pass

    def fit(self , X , y ):
        X = np.column_stack((np.ones(len(X)), X))
        self.theta = np.dot(np.linalg.inv(np.dot(X.T , X)) , np.dot(X.T , y ))
        print(self.theta)

    def predict(self,X_new):
        y_new = np.dot(X_new, self.theta)
        self.new = y_new
        
        return y_new
    def plot(self , X ,y):
        X_vals = np.linspace(X.min() , X.max() , 100)
        X_vals = np.column_stack((np.ones(len(X_vals)), X_vals))
        y_vals = np.dot(X_vals , self.theta)
        fig , ax = plt.subplots()
        ax.scatter(X,y, alpha = 0.5)
        ax.plot(X_vals , y_vals, color = 'red')
        plt.show()
        return fig

        