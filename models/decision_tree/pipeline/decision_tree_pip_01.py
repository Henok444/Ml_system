from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt 
from sklearn.pipeline import Pipeline


data = {
    "age": [22, 25, 47, 52],
    "salary": [30000, 35000, 90000, 100000],
    "bought": [0, 0, 1, 1]
}

df = pd.DataFrame(data)



X = df[["age","salary"]]
y = df['bought']



X_train , X_test , y_train , y_test = train_test_split(
    X , y , test_size= 0.2 , random_state= 42
)

pipline = Pipeline([
    ('model' , DecisionTreeClassifier(
        max_depth= None,
        criterion= "gini"
    )),

])

pipline.fit(X_train , y_train)
prediction = pipline.predict(X_test)
print(prediction)
print(y_test)
accuracy = accuracy_score(y_test , prediction)
print(accuracy)

plt.figure(figsize=(10,6))

plot_tree(
    pipline.named_steps['model'] ,
    feature_names= X.columns.tolist(),
    class_names = ["No" , "Yes"] , 
    filled = True 
)
plt.show()