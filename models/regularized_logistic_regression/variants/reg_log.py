import numpy as np
from utils.data_loader import load_classification_data_split_transformed
from sklearn.metrics import accuracy_score


class RegularizedLogisticRegressionGD:
    

    def __init__(self):
        pass

    def fit(self, X_train, y_train, alpha , epoch , lamda ):
        
        n , f = X_train.shape 
        self.theta = np.array([[0.0],[0.0]]) # matrix (2 , 1) # x matrix shape = (n,f(for this = 2))
        for i in range(epoch):


            z = np.dot(X_train , self.theta)   # matrix shape (n , 1)
            
            y_pred = 1 / (1 + np.exp(-z)) #matrix shape = n ,1

            error = y_pred - y_train  # n , 1 

            gradient = (1/n) * np.dot(X_train.T , error) # matrix shape ( 2 , 1 )

            penality_term = lamda * self.theta 

            penality_term[0] =  0 

            gradient += penality_term

            self.theta = self.theta - alpha * gradient



        
        # Convert probabilities to crisp 0 or 1 predictions
        predictions = (y_pred >= 0.5).astype(int)
        return print(f"Accuracy: {accuracy_score(y_train, predictions)}")


    

    def pridict(self , X_new):
        z = np.dot(X_new , self.theta)
        y_pred = 1 / (1 + np.exp(-z)) #matrix shape = n ,1
        y_new = (y_pred >= 0.5).astype(int)
        return y_new 
