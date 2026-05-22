# Experiment 01 
## model = logistic regression 
  ### from scratch
    for the training set i have used for exp_learning_rate 
    changing learning rate did not affect the accuracy after 0.1 
    and and more that 100 itration also wont change the accuracy of the model

that was because in data_loader the function add ones then scale , when scale after it add one because the std of the one column is zero so it squash the value to zero so all the values became zero so intercept term became zero always. i fixed it by adding one after scaling then the accuracy became equal with sklean model....
================================================================

