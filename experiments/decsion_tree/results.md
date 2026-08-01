#### exp_01_testing_my_model — from-scratch decision tree

Gini-based decision tree (max depth 3) implemented from scratch on
Social_Network_Ads (80/20 split, seed 42).

- **Test accuracy = 0.9**

Notes: `build_tree` requires a 1-D `y` — the loader returns `(n, 1)`, so the
tree now ravels the labels before splitting (leaves were previously a mix of
scalars and 1-element arrays, which broke `predict_batch`).
================================================================
