
# SVM using sklearn library

## Learning notes
- C controls the bias-variance trade-off: a small C gives high bias but low
  variance; a large C does the opposite.
- Default SVM accuracy was much higher than logistic regression on the same
  data.
- Kernel comparison on this data: poly < linear < rbf.

## Exp 01 — GridSearchCV tuning (`sklearn/exp_01_tuning_parameter_GridSearchCV.py`)
Breast cancer, 80/20 split, seed 42. Grid: C ∈ {0.1, 1, 10, 100},
gamma ∈ {1, 0.1, 0.01, 0.001}, rbf kernel, 5-fold CV.
- **Best params: C = 10, gamma = 0.01, kernel = rbf**
- Best CV accuracy = 0.9736
- **Test accuracy = 0.9825**

## Exp 02 — manual tuning (`sklearn/exp_02_tuning_parameter_self.py`)
Manual C sweep, rbf kernel, gamma = 0.01, same split as Exp 01.

| C    | Test accuracy |
|------|---------------|
| 0.1  | 0.9649        |
| 1    | 0.9649        |
| 10   | 0.9825        |
| 100  | 0.9561        |

C = 10 is the best manual choice, matching the GridSearchCV result. The
previous notes (fixed C = 10 trained 10 times, gamma sweep) came from an older
version of the script and no longer match its output.
================================================
