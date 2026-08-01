# Experiment 01 — logistic regression from scratch
## Learning rate and iterations
- Changing the learning rate did not affect accuracy after 0.1, and more than
  100 iterations also stopped changing the model accuracy.
- Root cause found: `data_loader` added the ones (intercept) column *before*
  scaling. Since the std of the ones column is zero, scaling squashed it to
  zero, so the intercept term was always zero. Fixed by adding the ones column
  *after* scaling — after the fix accuracy matched the sklearn model.

## Measured (from-scratch GD, Social_Network_Ads, lr = 0.01)
- 1 iteration → train acc 0.359
- ≥ 10 iterations → train acc 0.7875 (converged)
- Script-reported test accuracy: 0.7375 across all iteration counts

# Experiment 02 — sklearn baseline on breast cancer
`exp_02_breastcancer_data_test.py`: `StandardScaler + LogisticRegression`
pipeline on the breast cancer dataset (test split 20%, seed 42).
- **Test accuracy = 0.9737**
================================================================
