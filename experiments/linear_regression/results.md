# Linear Regression experiments

Dataset: `datasets/linear_regression/turbine_5yr_complex_data.csv` (features 2-3,
scaled), 80/20 split, `random_state=42`. Train R² reported on the train split.

## exp_01 — Normal Equation (`exp_01_testing_NE_model.py`)
Closed-form solution, no tuning.
- **Train R² = 0.9999**

## exp_02 — Batch GD learning rate sensitivity (`exp_02_learningrate_test_for_GD.py`)
Batch GD, 1000 epochs.
- lr = 0.001 → Train R² = **−0.055** (too slow, not converged)
- lr = 0.01  → Train R² = **0.1516**
- lr = 0.1   → Train R² = **0.1516** (converges to the same plateau as 0.01)

Conclusion: with plain scaled features, batch GD converges very slowly; the
normal equation is the reliable closed-form option for this data.

## exp_03 — Stochastic GD (`exp_03_testing_SGD_model.py`)
SGD, 1 epoch, alpha = 0.01 (no seed → varies run to run).
- **Train R² ≈ 0.13**

====================================================================
