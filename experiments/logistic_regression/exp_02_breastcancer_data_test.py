from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd 
from utils.plotting import plot
from sklearn.linear_model import LogisticRegression
import numpy as np 
from sklearn.metrics import r2_score

X , y   = load_breast_cancer( return_X_y= True)

X_train , X_test , y_train , y_test = train_test_split(
    X, y , test_size=0.2 , random_state=42
)
model = Pipeline([
    ("scale" , StandardScaler()),
    ("linear" , LogisticRegression())
])
model.fit(X_train,y_train)
y_new = model.predict(X_test)

w = model.named_steps['linear'].coef_
b = model.named_steps['linear'].intercept_

theta = np.column_stack((b , w))
accuracy = r2_score(y_test, y_new)
print(accuracy)

