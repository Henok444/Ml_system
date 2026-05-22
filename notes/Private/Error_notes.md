# Errors and how i solve them 

## Date = 4/23/26 


when trying to plot vs code the function dont return 
    # used fig, ax = plt.subplot 
still it doest show figure 
    # used plot.show()
ValueError: Expected 2D array, got 1D array instead:
    ## Use double brackets to keep it 2D from the start
        X = df[['your_feature_column']]
always X = df[[""]] #double[]

## Date = 4/22/26
fit in StandardScaler .fit always expect 2D
so when using iloc for access
    # df.iloc[:,[4]]
when use .fit x is transformed in to array so to access it use x[3,4]
but y is not transformed so it is still data frame to access it use x.iloc[4,5]
