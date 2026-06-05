from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from sklearn.datasets import load_iris

data = load_iris()

X = data.data
y = data.target

X_train , X_test , y_train , y_test = train_test_split(
    X , y , test_size=0.2 , random_state= 42
)

# base learner / grad boosting does not need base learner 




# ensemble 

grad = GradientBoostingClassifier(
    n_estimators=100 ,
    learning_rate = 0.1,
    max_depth= 3 , 
    random_state= 42



)

# train 

grad.fit(X_train , y_train)


# predict 

y_pred = grad.predict(X_test)

accuracy = accuracy_score(y_test , y_pred)

print(accuracy)
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance

result = permutation_importance(
    grad,
    X_test,
    y_test
)
import pandas as pd 
df = pd.DataFrame(result)
print(result)