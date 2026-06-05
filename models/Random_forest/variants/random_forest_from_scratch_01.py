import numpy as np 
from sklearn.tree import DecisionTreeClassifier

X = np.array([
    [2],
    [3],
    [10],
    [19]
])

y = np.array([0,0,1,1])

n_samples = len(X)
# bootstrap sampling 
indices = np.random.choice(
    n_samples,
    size = n_samples,
    replace = True

)
print(indices)

#now create a bootstrap data set

X_sample = X[indices]
y_sample = y[indices]

print (X_sample)

# now lets build a function that does all this

def bootstrap_sampling(X, y ):
    n_samples = len(X)
    indices = np.random.choice(
        n_samples , 
        size = n_samples ,
        replace=True
        
    )
    X_sample = X[indices]
    y_sample = y[indices]
    return X_sample , y_sample 
print(bootstrap_sampling(X,y))

# now how the learning is implemented 


n_trees = 10 
trees = []

for i in range(n_trees):

    X_boot, y_boot = bootstrap_sampling(X, y)

    tree = DecisionTreeClassifier()

    tree.fit(X_boot, y_boot)

    trees.append(tree)

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
    def fit(self ):
        self.trees = []
        for i in range(self.n_trees):

            # bootstrap data set 
            X_sample , y_sample = self.bootstrap_sampling()

            # create a tree 
            tree = DecisionTreeClassifier(
                max_depth= self.max_depth
            )
            #train tree 
            tree.fit(X_sample , y_sample)
            self.trees.append(tree)
    def pridict_sample(self, X):
        prediction = []
        for tree in self.trees:
            pred = tree.predict([X])[0]
            prediction.append(pred)
        value , counts = np.unique(
            prediction , 
            return_counts=True 
        )
        return value[np.argmx(counts)]
        

    
        
