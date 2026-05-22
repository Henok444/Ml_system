from models.linear_regression.variants.batch_gd import LinearRegressionGD
from utils.data_loader import load_data_split_transformed
from utils.plotting import plot
import numpy as np
from sklearn.metrics import r2_score


X , X_test , y , y_test = load_data_split_transformed("datasets/linear_regression/turbine_5yr_complex_data.csv" , 2,3)

model = LinearRegressionGD()

model.fit(X,y, lr = 0.001 , epochs = 1000)
t = np.column_stack((model.bias , model.weight))

y_new = model.predict(X)

plot(X,y,t,scaled=False)
print(f"error for Gd : {r2_score(y , y_new)}")
