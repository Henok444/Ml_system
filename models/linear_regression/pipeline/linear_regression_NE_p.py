from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from utils.data_loader import load_turbine
from utils.plotting import plot
import numpy as np

df = load_turbine()
X = df[['vibration_x']]
y = df['vibration_y']

X_train , X_test , y_train , y_test = train_test_split(
    X, y , test_size = 0.2 , random_state = 42
)
model =  Pipeline([
    ("scaled" , StandardScaler()),
    ('linear'  ,LinearRegression())
])
model.fit(X_train , y_train)
model.predict(X_test)

intercept = model.named_steps['linear'].intercept_
slope = model.named_steps['linear'].coef_
t = np.column_stack((intercept , slope))
plot(X,y, t , scaled = True)