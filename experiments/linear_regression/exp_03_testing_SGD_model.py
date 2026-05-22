from models.linear_regression.variants.stochastic_gd import LinearRegressionSGD
from utils.data_loader import load_data_split_transformed
from sklearn.metrics import r2_score 
from utils.plotting import plot
import numpy as np 

X_train , X_test , y_train , y_test = load_data_split_transformed("datasets/linear_regression/turbine_5yr_complex_data.csv" ,2 , 3 )

model = LinearRegressionSGD()

w , b = model.fit(X_train, y_train, 1, 0.01 )

t = np.column_stack((b,w))

y_new = model.predict(X_train)

print(f"error : {r2_score(y_train , y_new)}")

plot(X_train,y_train,t,scaled=False)



