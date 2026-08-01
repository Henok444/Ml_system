# Experiment 01 — regularized logistic regression (from scratch, L2)
## Measured (Social_Network_Ads, lr = 0.01, lambda = 1)
Test accuracy by iteration count:

| Iterations | Train acc | Test acc |
|-----------|-----------|----------|
| 1         | 0.3594    | 0.7375   |
| 10        | 0.7844    | 0.7375   |
| 100       | 0.7438    | 0.7125   |
| 1000+     | 0.6406    | 0.6500   |

After ~100 iterations the accuracy degrades — with this regularization term the
model drifts from the optimum as training continues. Compare with the
unregularized version, which stays at 0.7875 train / 0.7375 test.
================================================================
