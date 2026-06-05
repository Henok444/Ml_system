import numpy as np 

# algorithm 
"""
split data →
compute impurity →
keep the best split"""


X = np.array([2, 3, 10, 19,20,22,26,])
y = np.array([0, 0, 1, 1,0,0,1])

# what threshold should we try?


def gini(y_left , y_right):
    classes , counts_left = np.unique(y_left, return_counts= True)
    probability_left = counts_left / counts_left.sum()
    sq_left = probability_left**2
    gini_left =  1 - sq_left.sum()

    classes , counts_right = np.unique(y_right, return_counts= True)
    probability_right = counts_right / counts_right.sum()
    sq_right = probability_right**2
    gini_right =  1 - sq_right.sum()

    n = len(y_left) + len(y_right)
    w_le = len(y_left) / n
    w_ri = len(y_right) / n 
    gini_weighted_avg = (w_le * gini_left) + (w_ri*gini_right)
    return gini_weighted_avg 
def best_split(X,y):

    best_treshold = None 
    best_score = float("inf")
    best_left = []
    best_right = []
    thresholds = np.unique(X)
        
    for threshold in thresholds :
        t = threshold
        # split the data 
        left_mask = X < t
        right_mask = X >= t

        X_left = X[left_mask]
        X_right = X[right_mask]

        y_left = y[left_mask]
        y_right = y[right_mask]
        if len(y_left) == 0 or len(y_right) == 0:
            continue

        # score 
        new_score = gini(y_left , y_right)
        if new_score < best_score:

            best_treshold = threshold
            best_score = new_score 

            best_left = y_left 
            best_right = y_right
    return best_treshold

def most_common_lable(y):
    value , counts = np.unique(y , return_counts = True)
    index = np.argmax(counts)
    return value[index]

"""
def predict(y):
    left_prediction = most_common_lable(best_left)
    right_prediction = most_common_lable(best_right)
    if y < best_treshold:
        return left_prediction 
    else:
        return right_prediction 

print(best_threshold)
print(best_left)
print(best_right)
"""

def build_tree(X , y , depth = 0 , maxdepth = 3):
    #stoping criteria 
    if len(np.unique(y)) == 1:
        return y[0]
    if depth >= maxdepth :
        prediction = most_common_lable(y)
        print("max depth leaf : " , prediction)
        return prediction
    # finding the threshold 
    threshold = best_split(X , y)
    # split the data 
    left_mask = X < threshold
    right_mask = X >= threshold

    X_left = X[left_mask]
    y_left = y[left_mask]

    X_right = X[right_mask]
    y_right = y[right_mask]

    # create a subtree


    left_subtree = build_tree(X_left , y_left , depth+1 , maxdepth )
    right_subtree = build_tree(X_right , y_right , depth+1 , maxdepth)
    
    return print ({
        "threshold":threshold ,
        "left" : left_subtree , 
        "right" : right_subtree
    })
build_tree(X , y)



    
