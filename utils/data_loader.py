import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
import numpy as np 

def load_data(location, column_x , column_y ):
    data = pd.read_csv(location)
    df = pd.DataFrame(data)
    X = df.iloc[:,[column_x]]
    y = df.iloc[:,column_y]
    return X , y 
def load_data_split(location,column_x , column_y):
    data = pd.read_csv(location)
    df = pd.DataFrame(data)
    X = df.iloc[[column_x]]
    y = df.iloc[column_y]

    X_train , X_test , y_train ,y_test = train_test_split(
        X , y , test_size = 0.2 , random_state = 42
    )
    return X_train , X_test , y_train ,y_test
def load_data_transformed(location, column_x , column_y ):

    data = pd.read_csv(location)
    df = pd.DataFrame(data)
    X = df.iloc[[column_x]]
    y = df.iloc[column_y]

    scale = StandardScaler()
    scale.fit(X)
    X_train = scale.transform(X)
    return X , y 

def load_data_split_transformed(location, column_x , column_y ):
    data = pd.read_csv(location)
    df = pd.DataFrame(data)
    X = df.iloc[:,[column_x]]
    y = df.iloc[:,column_y]

    X_train , X_test , y_train ,y_test = train_test_split(
        X , y , test_size = 0.2 , random_state = 42
    )

    scale = StandardScaler()
    scale.fit(X_train)
    X_train = scale.transform(X_train)
    X_test = scale.transform(X_test)

    return X_train , X_test , y_train ,y_test
def load_classification_data_split_transformed(location, column_x,column_y, change_y ,  string_to_be_one  ,string_to_be_zero ):
    data = pd.read_csv(location)
    data_c = data.copy()
    df = pd.DataFrame(data_c)
    X = df.iloc[:,[column_x]]
    y = df.iloc[:,[column_y]].values

      
    if change_y:
        y = y.replace({string_to_be_one : 0 , string_to_be_zero : 1 })

    else: 
        pass 



    X_train ,X_test , y_train , y_test = train_test_split(
        X , y , test_size=0.2 , random_state=42 
    )
    scale = StandardScaler()
    scale.fit(X_train)
    X_train = scale.transform(X_train)
    X_test = scale.transform(X_test)  

    X_train = np.column_stack((np.ones(len(X_train)), X_train))
    X_test = np.column_stack((np.ones(len(X_test)), X_test))

    return   X_train ,X_test , y_train , y_test                           
def load_classification_data_split_transformed_for_sklearn(location, column_x,column_y, change_y ,  string_to_be_one  ,string_to_be_zero ):
    data = pd.read_csv(location)
    data_c = data.copy()
    df = pd.DataFrame(data_c)
    X = df.iloc[:,[column_x]]
    y = df.iloc[:,column_y]

    X = np.column_stack((np.ones(len(X)),X))  
    if change_y:
        y = y.replace({string_to_be_one : 1 , string_to_be_zero : 2 })

    else: 
        pass 



    X_train ,X_test , y_train , y_test = train_test_split(
        X , y , test_size=0.2 , random_state=42 
    )
    scale = StandardScaler()
    scale.fit(X_train)
    X_train = scale.transform(X_train)
    X_test = scale.transform(X_test)  


    return   X_train ,X_test , y_train , y_test 