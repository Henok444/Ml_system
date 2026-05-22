new syntax i have learned

np.linalg.inv() -- inverse a matrix
fig, ax = plt.subplots() -- in function to return plot 
np.column_stack((x,y)) -- to create 2D matrix from 2 1D arrays
in train_test_split(return_split = True)
X = df.iloc[:, [3]]  # Returns a 2D DataFrame (with one column)

 Replace "string_a" with 0 and "string_b" with 1
df.iloc[:, 0] = column_to_change.replace({'string_a': 0, 'string_b': 1})
======
from sklearn.metrics import accuracy_score
- Convert probabilities to crisp 0 or 1 predictions
predictions = (y_pred >= 0.5).astype(int)
print(f"Accuracy: {accuracy_score(y_train, predictions)}")