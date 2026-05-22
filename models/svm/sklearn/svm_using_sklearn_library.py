from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report , accuracy_score
from sklearn.model_selection import train_test_split


# split and normalize


X , y = load_breast_cancer(return_X_y=True)

X_train , X_test , y_train , y_test = train_test_split(
    X, y , test_size=0.2 , random_state= 42
)

scale = StandardScaler()
scale.fit(X_train)
scale.transform(X_train)
scale.transform(X_test)


#model 


model = SVC(
    C= 100.0 , # this control the trade off 
                # 
    kernel='rbf',
    degree= 3,
    coef0= 0.0,
    probability=False

)
model.fit(X_train , y_train)
y_pred = model.predict(X_test)

print(f"classification_report = {classification_report(y_test , y_pred)}" )

