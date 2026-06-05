import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

'''
for each tree:
    create bootstrap sample
    train decision tree
    store tree

during prediction:
    get predictions from all trees
    majority vote
    '''

class RandomForest:
    def __init__(self , n_trees = 10 , max_depth = 3):
        self.n_trees = n_trees 
        self.max_depth = max_depth 
        self.trees = []
    def bootstrap_sampling(self , X, y ):
        n_samples = len(X)
        indices = np.random.choice(
            n_samples , 
            size = n_samples ,
            replace=True
            
        )
        return X[indices] , y[indices]
    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_trees):

            # bootstrap data set 
            X_sample , y_sample = self.bootstrap_sampling(X,y)

            # create a tree 
            tree = DecisionTreeClassifier(
                max_depth= self.max_depth
            )
            #train tree 
            tree.fit(X_sample , y_sample)
            self.trees.append(tree)
    def predict_sample(self, X):
        prediction = []
        for tree in self.trees:
            pred = tree.predict([X])[0]
            prediction.append(pred)
        value , counts = np.unique(
            prediction , 
            return_counts=True 
        )
        return value[np.argmax(counts)]
    def predict(self, X):

        predictions = []

        for x in X:

            pred = self.predict_sample(x)

            predictions.append(pred)

        return np.array(predictions)
        

    
model = RandomForest(
    n_trees=10,
    max_depth=3
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(
    y_test,
    predictions
)

print(accuracy)
print(model.trees)