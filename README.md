# Ml_system

A hands-on machine learning lab: classic algorithms implemented **from scratch**,
then re-implemented with **scikit-learn** and packaged into **pipelines** — with
per-algorithm experiments, notes, and results.

The goal of this repo is the learning journey: each model folder shows the same
algorithm at three levels of maturity.

## Repository Structure

```
models/                    Algorithms at three levels: from-scratch → sklearn → pipeline
  linear_regression/       Batch GD, Normal Equation, Stochastic GD
  logistic_regression/     Gradient descent, binary cross-entropy
  regularized_logistic_regression/  L2-regularized GD
  decision_tree/           Split quality, thresholds (level_0 → level_1 → simple)
  Random_forest/           Bagging + randomized feature selection
  polynomial_regression/    Degree selection & feature engineering
  KNN/                     Distance-based classification
  svm/                     Hinge loss, kernels (notes + sklearn + pipeline)
  bagging/                 Bootstrap aggregation
  boosting/                AdaBoost + Gradient Boosting
  XG_boost/                Gradient-boosted trees
experiments/               Per-algorithm experiment scripts + results.md
utils/                     Shared data loading, metrics, and plotting helpers
configs/                   Config files
api/                       Crop recommendation REST API (FastAPI)
notebooks/                 EDA and project notebooks
reports/                   Project reports and visualizations
```

## Implemented Models

| Model | From scratch | scikit-learn | Pipeline | Notes |
|-------|:---:|:---:|:---:|-------|
| Linear Regression | ✅ (batch GD, normal eq., SGD) | ✅ | ✅ | ✅ |
| Logistic Regression | ✅ (GD, BCE) | ✅ | ✅ | ✅ |
| Regularized Logistic Regression | ✅ (L2) | ✅ | ✅ | ✅ |
| Decision Tree | ✅ (3 levels) | ✅ | ✅ | ✅ |
| Random Forest | ✅ | ✅ | — | ✅ |
| Polynomial Regression | ✅ | — | ✅ | ✅ |
| KNN | — | ✅ | — | ✅ |
| SVM | — | ✅ | ✅ | ✅ |
| Bagging | ✅ | — | — | — |
| AdaBoost / Gradient Boosting | ✅ | — | — | — |
| XGBoost | — | ✅ | — | — |

## Selected Experiment Results

| Experiment | Setup | Result |
|-----------|-------|--------|
| Linear Regression — learning rate sensitivity (batch GD, turbine) | lr ∈ {0.001, 0.01, 0.1}, 1000 epochs | 0.001 too slow (R² −0.055); 0.01/0.1 converge to R² 0.15; normal equation reaches R² 0.9999 |
| Logistic Regression — from scratch vs sklearn | GD with intercept scaling bug found & fixed | From-scratch train acc 0.7875; sklearn baseline 0.9737 (breast cancer) |
| Polynomial Regression — degree selection | degree sweep 2–19 | Best degree: 8 |
| Decision Tree — from-scratch baseline | own splitter (gini, depth 3) | Test accuracy 0.9 — `experiments/decsion_tree/results.md` |
| SVM — hyperparameter tuning | GridSearchCV vs manual tuning | Best C=10, gamma=0.01, rbf — test 0.9825 — `experiments/svm/sklearn/` |

## Featured Project: Ensemble Learning for Precision Agriculture

Crop recommendation on 2,200 observations × 22 crop classes with Random Forest
and XGBoost (GridSearchCV-tuned), evaluated with accuracy/precision/recall/F1
and interpreted with SHAP.

- Full project repo: [crop-recommendation-ensemble](https://github.com/Henok444/crop-recommendation-ensemble)
- Production API repo: [crop-recommendation-api](https://github.com/Henok444/crop-recommendation-api)
- Notebook: `notebooks/Practical_Ensemble_Learning_Precision_Agriculture.ipynb`
- Report: [Practical_Ensemble_Learning_Precision_Agriculture.pdf](https://github.com/Henok444/Ml_system/releases/tag/reports-v1) (GitHub Release)

## Setup

```
pip install -r requirements.txt
```

Run any experiment script directly, e.g.:

```
python experiments/linear_regression/exp_01_testing_NE_model.py
```

The experiment scripts expect a `datasets/` folder next to `utils/` (gitignored
— the data files come from public sources such as Kaggle/UCI). Without it the
scripts cannot load data; the results recorded in each `experiments/*/results.md`
come from real runs with those datasets.

## Built With

Python · NumPy · Pandas · Matplotlib · Scikit-learn · XGBoost · SHAP · FastAPI
