from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from utils.data_loader import load_classification_data_split_transformed_for_sklearn
from sklearn.metrics import accuracy_score



X_train , X_test , y_train , y_test = load_classification_data_split_transformed_for_sklearn("datasets/classification/raw/Social_Network_Ads.csv" , 1, 2, change_y= False ,string_to_be_one= 0 , string_to_be_zero=0 )

model = Pipeline([
    ("linear"  , LogisticRegression(penalty= 'l2' , C= 2.0)),
    
])
model.fit(X_train , y_train)
y_new = model.predict(X_test)

print(f"Accuracy :{accuracy_score(y_test , y_new )}")
