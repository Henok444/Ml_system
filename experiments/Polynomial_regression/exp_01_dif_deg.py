from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.data_loader import load_data
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np 

X, y = load_data("datasets/linear_regression/DailyDelhiClimateTest.csv" , 2,3)


X_train , X_test , y_train , y_test = train_test_split(
 X,y , test_size = 0.2 , random_state=42
)
score_train = []
score_test = []
degrees = []
for i in range(2,20):    
    model = Pipeline([
        ('scal' , StandardScaler()),
        ('poly' , PolynomialFeatures(degree=i)),
        ('linear' , LinearRegression())

    ])
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    score_train.append(r2_score(y_train , y_pred ))
    score_test.append(r2_score(y_test , y_test_pred))
    degrees.append(i)

fig , ax = plt.subplots()
ax.plot(range(2,20) ,score_train, color = 'red' )
ax.scatter(range(2,20) ,score_train, color = 'red' )
ax.plot(range(2,20) ,score_train, color = 'red' )
ax.scatter(range(2,20) ,score_test, color = 'blue' )


print(f"{degrees[np.argmax(score_test)]}")
plt.show()

