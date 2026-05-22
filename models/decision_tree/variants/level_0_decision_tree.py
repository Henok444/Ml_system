import numpy as np 

import numpy as np

X = np.array([2, 3, 10, 19])
y = np.array([0, 0, 1, 1])

#we want the computer to discover a rule 

threshold = 10 

# create a left and right group 
left_mask = X < threshold     # this creat a list of boolean
right_mask = X >= threshold 

x_left = X[left_mask]
x_right = X[right_mask]

y_left = y[left_mask]
y_right = y[right_mask]  

print("Left X:", x_left)
print("Right X:", x_right)

print("Left y:", y_left)
print("Right y:", y_right)


# now make a prediction 

def predict(x):
    if x < 10:
        return 0 
    else:
        return 1 
    
answer = predict(11)
print(answer)

"""
What i Just Built

i already built the CORE of a decision tree:

choose a threshold,
split the data,
separate classes,
predict using rules.

Everything else in decision trees is just automating:

how thresholds are chosen,
how many splits happen,
how deep the tree grows.
"""

classes , counts = np.unique(y , return_counts = True)
print(classes)
print(counts)

probability = counts / counts.sum()
print(probability)

squared = probability**2
gini = 1 - squared.sum()
print(gini)

#creating a function 
def gini(y):
    classes , counts = np.unique(y , return_counts = True) # return a list btw
    probability = counts / counts.sum()
    squared = probability**2
    return 1-squared.sum()
print(gini(y_left))
print(gini(y_right))

# weighted average of the right side split and the left side split
w_le = len(y_left) / len(y)
w_ri = len(y_right) / len(y)

gini_avarage = w_le * gini(y_left) + w_ri * gini(y_right)



