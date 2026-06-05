from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_iris

data = load_iris()

X = data.data
y = data.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# base model 
tree = DecisionTreeClassifier(
    max_depth=3
)

bagging = BaggingClassifier(

    estimator=tree,

    n_estimators=50,

    random_state=42
)

bagging.fit(X_train, y_train)

predictions = bagging.predict(X_test)



accuracy = accuracy_score(
    y_test,
    predictions
)

print(accuracy)