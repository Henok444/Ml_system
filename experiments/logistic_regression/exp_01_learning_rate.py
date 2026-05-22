from models.logistic_regression.variants.log_gd import LogisticRegressionGD
from utils.data_loader import load_classification_data_split_transformed
import numpy as np 
from sklearn.metrics import accuracy_score










X_train , X_test , y_train ,y_test = load_classification_data_split_transformed("datasets/classification/raw/Social_Network_Ads.csv" , 1 , 2 , change_y=False , string_to_be_one= False, string_to_be_zero= False)


model = LogisticRegressionGD()

   

for i in range(6):

    model.fit(X_train, y_train , 0.01 , 1*(10**i))


    y_new = model.pridict(X_test)
    print(f"Accuracy for {1*(10**i)} and alpha of {0.01}: {accuracy_score(y_test, y_new)}")