import numpy as np 

for i in range(2,degree+1):
    X_fet = np.power(X, i)
    X_new = np.column_stack((X,X_new))
    
