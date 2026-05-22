from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report , accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np




X , y = load_breast_cancer(return_X_y=True)

X_train , X_test , y_train , y_test = train_test_split(
    X, y , test_size=0.2 , random_state= 42
)

scale = StandardScaler()
scale.fit(X_train)
scale.transform(X_train)
scale.transform(X_test)
C_list = []
for i in range(10):
    
    model = SVC(
        C= 10 ,
        kernel='rbf',
        degree= 3,
        coef0= 0.0,
        probability=False ,
        gamma= 0.01

    )
    model.fit(X_train , y_train)
    y_pred = model.predict(X_test)

    #print(f"classification_report = {classification_report(y_test , y_pred)}" )
    g = accuracy_score(y_test , y_pred)
    print(accuracy_score(y_test , y_pred))
    C_list.append(g)
print(1 / float(10.0**(np.argmax(C_list) - 5)), C_list)