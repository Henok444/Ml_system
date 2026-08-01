from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


X , y = load_breast_cancer(return_X_y=True)

X_train , X_test , y_train , y_test = train_test_split(
    X, y , test_size=0.2 , random_state= 42
)

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

for C in [0.1, 1, 10, 100]:
    model = SVC(
        C= C ,
        kernel='rbf',
        gamma= 0.01
    )
    model.fit(X_train , y_train)
    y_pred = model.predict(X_test)
    print(f"C={C}: accuracy {accuracy_score(y_test , y_pred):.4f}")
