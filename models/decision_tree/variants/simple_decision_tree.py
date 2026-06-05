import numpy as np
import pandas as pd 


class DecisionTree():
    def __init__(self):

        pass
    def gini(self , y):
        classes , counts_left = np.unique(y, return_counts= True)
        probability = counts_left / counts_left.sum()
        sq = probability**2
        gini =  1 - sq.sum()
        return gini
    def split_gini(self , y_left , y_right):
        gini_left =  self.gini(y_left)
        gini_right =  self.gini(y_right)

        n = len(y_left) + len(y_right)
        w_le = len(y_left) / n
        w_ri = len(y_right) / n 
        gini_weighted_avg = (w_le * gini_left) + (w_ri*gini_right)
        return gini_weighted_avg 

    def best_split(self , X , y ):
        y= np.asarray(y)
        n_features = X.shape[1]

        best_feature = None 
        best_threshold = None 
        best_score = float("inf")

        for feature in range(n_features):
            column = X[:,feature]
            thresholds = np.unique(column) + 0.5
            for threshold in thresholds:
                left_mask = column < threshold
                right_mask = column >= threshold 
                y_left = y[left_mask]
                y_right = y[right_mask]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                score = self.split_gini(y_left , y_right)
                if score < best_score:
                    best_score = score
                    best_threshold = threshold
                    best_feature = feature

        return (best_feature , best_threshold)
    
    def most_common_label(self , y):
        value , counts = np.unique(y , return_counts = True)
        index = np.argmax(counts)
        return value[index]
    def build_tree(self,  X , y , depth = 0 , maxdepth=3):
            #stoping criteria 
        if len(np.unique(y)) == 1:
            return y[0]
        if depth >= maxdepth :
            prediction =self.most_common_label(y)
            print("max depth leaf : " , prediction)
            return prediction
        # finding the threshold 
        feature , threshold = self.best_split(X , y)
        if feature is None:
            return self.most_common_label(y)


        # split the data 
        feature_column = X[:,feature]
        left_mask = feature_column < threshold
        right_mask = feature_column >= threshold
        X_left = X[left_mask]
        X_right = X[right_mask]
        y_left = y[left_mask]
        y_right = y[right_mask]

        # create a subtree


        left_subtree = self.build_tree(X_left , y_left , depth+1 , maxdepth )
        right_subtree = self.build_tree(X_right , y_right , depth+1 , maxdepth)
        
        return {
            "feature" : feature,
            "threshold":threshold ,
            "left" : left_subtree , 
            "right" : right_subtree
        }
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)
    def predict(self, x):
        # 1. Define an inner recursive function that tracks the current node location
        def _traverse(node, row):
            # Base Case: If the node is a leaf value (0 or 1), return it
            if not isinstance(node, dict):
                return node
                
            # Extract split parameters from the current node
            feature = node["feature"]
            threshold = node["threshold"]
            
            # Extract sample value safely for Pandas rows or numpy arrays
            val = row.iloc[feature] if hasattr(row, "iloc") else row[feature]
            
            # Recursively descend down the tree passing the correct sub-node
            if val < threshold:
                return _traverse(node["left"], row)
            else:
                return _traverse(node["right"], row)

            # 2. Kick off the recursion starting at the root tree
            # Change 'self.tree' to whatever variable name holds your dictionary inside your class
        return _traverse(self.tree, x)

    def predict_batch(self, X):
        predictions = []
        for i in range(len(X)):
            # Handle both Pandas DataFrames (.iloc) and Numpy arrays safely
            row = X.iloc[i] if hasattr(X, "iloc") else X[i]
            predictions.append(self.predict(row))
        return np.array(predictions)

    

        
    
                

    

