
Immediately after running experiment.

Do NOT delay. If you delay, you lose insight.
==============================================================================

Experiment 02 : Learning Rate Sensitivity for GD

Setup:
- Model: Batch GD
- Dataset: Housing

Tested:
- lr = 0.001, 0.01, 0.1

Results:
- 0.001 → very slow
- 0.01 → stable
- 0.1 → diverged

Conclusion:
Optimal lr ~ 0.01

Insight:
High learning rate causes instability
====================================================================
exp_03_test_sgd


