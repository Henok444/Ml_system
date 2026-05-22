from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score

# data
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# pipeline (very important)


pipe = Pipeline([
    ("scale", StandardScaler()),
    ("svm", SVC())
])

params = {
    "svm__C":[0.1,1,10,100],
    "svm__gamma":[1,0.1,0.01,0.001],
    "svm__kernel":["rbf"]
}

grid = GridSearchCV(
    pipe,
    params,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train,y_train)

print(grid.best_params_)
model = SVC(
    C = 10 , 
    kernel= 'poly',
    degree= 4 ,
    gamma= 0.01,
    coef0= 0.00 , 
)
model.fit(X_train , y_train )
y_new = model.predict(X_test)
print(r2_score(y_test , y_new ), classification_report(y_test , y_new))
print(model.n_support_
)


