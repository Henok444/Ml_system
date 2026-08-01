from models.linear_regression.variants.normal_equation import LinearRegressionNE
from utils.data_loader import load_data_split_transformed
from sklearn.metrics import r2_score
import numpy as np 
import pandas as pd 
X_train , X_test , y_train, y_test = load_data_split_transformed("datasets/linear_regression/turbine_5yr_complex_data.csv", 3,4 )


model = LinearRegressionNE()

model.fit(X_train,y_train)

y_new = model.predict(np.column_stack((np.ones(len(X_train)), X_train)))
print(f"Train R2: {r2_score(y_train, y_new)}")

model.plot(X_train , y_train)
