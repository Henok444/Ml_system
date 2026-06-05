first try to build the model from scratch,

the algorithm i learned so far (for a data with only one feature)
    first we have to build a function that takes our data and return a rule in a dictionary format 
        fist check the general impurity of our data whether we shall split it or not     
        first it find the best splitting position of row.
            build a Gini impurity function. for testing each split impurity.
            build a function that try all possible splitting threshod and give you the best treshold by comparing there Gini impurity.
        then split the data to left and right. 
        then repeat the same for same process for both left and right split.
        then return the tree , all the recursive function also return there dictionary.
    now after we found the rule tree , we predict.
        first check if the tree is dictionary.
        then extract what the treshold is for the tree 
        if the new x is less than this threshold then prdict again using the treshold inside the left (in which it contain it our threshold). 
        do the same to the right if X > threshold. 
        in the leaf node left and right contain 0 or 1 not another tree. So the predict function return 0 0r 1
      
