When to write here
While coding the model
When something doesn’t work
When you fix something
==============================================================================

Model: Linear Regression

Core Idea:
Fit a linear function to minimize error

Loss Function:
MSE

Optimization Methods:
- Normal Equation
- Gradient Descent (Batch, SGD)

Key Observations:
- Sensitive to feature scaling
- Learning rate affects convergence

Implementation Notes:
- Initialize weights to zero
- Update using gradient

Problems Faced:
- Divergence at high learning rate

Fixes:
- Reduced learning rate
==========================================================
model = linear regression 
type = sk learn 

first you split the data then fit xtrain only then transform xtrain and x test , dont transform or fit y values
===============================


 Assuming your pipeline is named 'pipe' and the model step was named 'regressor'


