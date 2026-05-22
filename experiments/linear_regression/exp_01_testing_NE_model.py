from models.linear_regression.variants.normal_equation import LinearRegressionNE
from utils.data_loader import load_data_split_transformed
import pandas as pd 
X_train , X_test , y_train, y_test = load_data_split_transformed("datasets/linear_regression/turbine_5yr_complex_data.csv", 3,4 )


model = LinearRegressionNE()

model.fit(X_train,y_train)

model.plot(X_train , y_train)
