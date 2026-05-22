from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.data_loader import load_data

X, y = load_data("datasets/linear_regression/DailyDelhiClimateTest.csv" , 2,3)


X_train , X_test , y_train , y_test = train_test_split(
 X,y , test_size = 0.2 , random_state=42
)

model = Pipeline([
    ('scal' , StandardScaler()),
    ('poly' , PolynomialFeatures()),
    ('linear' , LinearRegression())

])
model.fit(X_train, y_train)
model.predict(X_train)

print(model.named_steps['linear'].coef_)