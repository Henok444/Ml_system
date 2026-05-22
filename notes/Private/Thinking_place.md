date : 4/23/26
## logistic Regression bulding from scratch 

"""
alg

recive data 
    -data_loader 

    create a function

    add ones column  to x  first position # done 
    change a value of y into 1 and 0 and 
            Replace "string_a" with 0 and "string_b" with 1
            df.iloc[:, 0] = column_to_change.replace({'string_a': 0, 'string_b': 1})

    do proper split and scaling then  

      use class add one in the feature eng class to add one column then return the value # another time just do it in the data_loader function.

    

feature engi

create a class separetly that can add one a column that have a value of one for the bias terms

#### done 


learn 

X, y 

y_pred = 1 / 1 - np.exp(X@theta) = h(x)

error = y - y_pred



gradient = 1 / m (error@X.T )

theta = theta - alpha*gradient






"""
