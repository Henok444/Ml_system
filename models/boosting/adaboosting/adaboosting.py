from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data  = load_iris()
X = data.data 
y = data.target 

X_train , X_test , y_train , y_test = train_test_split(
    X, y , test_size= 0.2 , random_state= 42
)

#Create Weak Learner

stump = DecisionTreeClassifier(
    max_depth = 1
)

ada = AdaBoostClassifier(
    estimator=stump,
    n_estimators= 50 ,
    learning_rate= 1.0 ,
    random_state = 42

)

ada.fit(X_train, y_train)
prediction = ada.predict(X_test)

accuracy = accuracy_score(
    y_test,
    prediction 
)
print(accuracy)


