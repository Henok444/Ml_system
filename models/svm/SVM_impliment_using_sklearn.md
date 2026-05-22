## implimenting SVC
To learn SVM in practice, treat sklearn.svm as several related tools, not one algorithm.

Main estimators in scikit-learn:

    SVC — classification (most common)
    LinearSVC — faster linear SVM for large data
    SVR — support vector regression
    LinearSVR — faster linear regression version
    OneClassSVM — anomaly detection

### SVC

SVC(
  C=1.0,
  kernel='rbf',
  gamma='scale',
  degree=3,
  coef0=0.0,
  probability=False
)

1) C (regularization)

Most important parameter.

SVC(C=0.1)
SVC(C=10)


Small C:

wider margin
more regularization
simpler model
may underfit

Large C:

tighter margin 
tries harder to classify all points
can overfit

Typical values to test:

[0.01,0.1,1,10,100]

2) kernels; 

a) Linear:

SVC(kernel="linear")


Use when data approximately linearly separable. 

b) RBF 
SVC(kernel="rbf")
Most common.
use it when the data points are linearly inseparable. so it transform it to higher dimension. 

c) Polynomial 
SVC(kernel="poly", degree=3)
to fit also transform the values of x 

3) gamma (for poly and rbf)

Important for RBF/poly.
SVC(gamma=0.01)
Kernel influence radius.
Large gamma:
    very local
    complex boundary
    overfitting risk
Small gamma:
    smoother boundary
    Common values:

[1,0.1,0.01,0.001]

4) degree

Only for polynomial kernel.
SVC(kernel="poly", degree=4)
Controls polynomial complexity.
Usually low degrees.

5) probability=True

SVC(probability=True)
Enables:
model.predict_proba(X)
Useful, but slower.
Hyperparameter tuning (critical)

#### tunning C and gamma together 

from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("svm", SVC())
])

params = {
    "svm__C":[0.1,1,10,100],
    "svm__gamma":[1,0.1,0.01,0.001],
    "svm__kernel":["rbf"]
}

grid = GridSearchCV(
    pipe,
    params,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train,y_train)

print(grid.best_params_)
This is standard SVM workflow.
Honestly:
Untuned SVM is often mediocre.
Tuned SVM can be excellent.
Very important attributes after fitting
model.named_steps["svm"].support_vectors_
Actual support vectors.
Number of support vectors:
model.named_steps["svm"].n_support_
Dual coefficients:
model.named_steps["svm"].dual_coef_
Intercept:
model.named_steps["svm"].intercept_
For linear kernel:
model.named_steps["svm"].coef_
weights w.
Inspecting support vectors is useful because it connects code to theory.

LinearSVC vs SVC
People confuse these.

SVC(kernel='linear') vs LinearSVC()

Not identical.
Use LinearSVC for large/high-dimensional data:

"""
    from sklearn.svm import LinearSVC

    model = Pipeline([
        ("scale",StandardScaler()),
        ("svm",LinearSVC(C=1))
    ])

"""

Often great for text classification.
Kernel SVM can be expensive.
Training roughly scales badly as data grows.