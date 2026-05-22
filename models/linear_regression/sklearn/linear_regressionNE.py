from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from utils.data_loader import load_turbine
from utils.plotting import plot
import numpy as np 


df = load_turbine() 

X = df[['vibration_x']]
y = df['vibration_y']



X_train , X_test , y_train , y_test = train_test_split(X,y , test_size= 0.2, random_state = 42 )
stand = StandardScaler()
stand.fit(X_train)
X_train = stand.transform(X_train)
X_test = stand.transform(X_test)

model = LinearRegression()
model.fit(X_train,y_train)

model.predict(X_train)
model.predict(X_test)

t = np.column_stack((model.intercept_ , model.coef_))

plot(X,y,t, scaled= True)
