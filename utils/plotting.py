import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.preprocessing import StandardScaler



def plot(X , y, theta ,scaled):

    if scaled:
        scale = StandardScaler()
        scale.fit(X)
        X = scale.transform(X)
        Xvals = np.linspace(X.min(), X.max() , 100)
        Xvals_2D = np.column_stack((np.ones(100),Xvals))
        yvals = np.dot(Xvals_2D , theta.T)
        
        

        fig , ax = plt.subplots()
        ax.scatter(X,y , alpha = 0.5)
        ax.plot(Xvals , yvals , color = 'red')
        plt.show()
    else:
        Xvals = np.linspace(X.min(), X.max() , 100)
        Xvals_2D = np.column_stack((np.ones(100),Xvals))
        yvals = np.dot(Xvals_2D , theta.T)

        fig , ax = plt.subplots()
        ax.scatter(X,y , alpha = 0.5)
        ax.plot(Xvals , yvals , color = 'red')
        plt.show()
    return ax


