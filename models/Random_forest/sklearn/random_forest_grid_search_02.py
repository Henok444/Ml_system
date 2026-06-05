import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

from sklearn.model_selection import GridSearchCV

data = load_iris()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
pipeline = Pipeline([

    ("imputer", SimpleImputer(strategy="mean")),

    ("scaler", StandardScaler()),

    ("model", RandomForestClassifier(

        random_state=42
    ))
])

param_grid = {

    "model__n_estimators": [50, 100],

    "model__max_depth": [3, 5, 10],

    "model__min_samples_split": [2, 5],

    "model__max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(

    estimator=pipeline,

    param_grid=param_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)
best_model = grid_search.best_estimator_

predictions = best_model.predict(X_test)
accuracy = accuracy_score(
    y_test,
    predictions
)

print(accuracy)