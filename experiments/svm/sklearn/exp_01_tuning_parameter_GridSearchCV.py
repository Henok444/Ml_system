from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

print("Best params:", grid.best_params_)
print(f"Best CV accuracy: {grid.best_score_:.4f}")
y_new = grid.best_estimator_.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_new):.4f}")


