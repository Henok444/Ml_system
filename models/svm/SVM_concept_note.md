## SVM Notes 

what is svm?

svm stands for support vector machine, its supervised machine learning model used for mainly classification but also for regression.
svm asks can i separate classes whith the best possible boundary? not just any boundary but with the largest margin?
suppose we have two classes R and B 

Bad boundary:         Better boundary:        SVM boundary:

R R | B B             R |   B B               R     |     B
R R | B B             R |   B B               R     |     B
                       ↑ narrow margin         ← wide margin →
svm chose a boundary with the maximam margin. 
the decision boundary for linear SVM is w^T.X + bias = 0 
w = weights (orientation of boundary)
b = bias (position)
x = input point
Points satisfying

wTx+b>0------------belong to one class, and

wTx+b<0------------blongs to other

what is margin then?
it is the distance from the boundary to the nearest training point. margin = 2 / ||W||   
svm is trying to maximize this. 

what is support vectors?

only the closest critical point determine the boundary. those points are support vectors.
move distant point around boundary barely changes.
move support vectors change the boundary.they support the hyperplane.
unlike logistic regression the SVM mainly depend on these boundary examples 

SoftMarigin SVM (real world SVM)

perfect separation is rare 

so as there is regularization for logistic regression,(penalizing the parameters to control the bias and variance trade offs) for svms we introduce slack variable , +Ci∑​ξi​ term for the optimized term. 
Interpretation:

First term → large margin
Second term → penalize classification errors

C controls tradeoff:
large c cares about few errors , thighter boundary and may overfit .
smaller c allows a mistakes , wider margin and stronger regularization. 
 
## non linnear svms ----kernal trics

what if data is not linearly separable?
map data into higher dimension then maybe in transformed space it becames separable.
instead of explicitly transforming features SVM uses kernals.

common kernels;

A) Linear kernel;
    just ordinary linear SVM

B) Polonomial kernel 
    Create curve boundaries 

C) RBF/gaussian kernel(very common )
    gamma controls locality 
        high gamma very wiggly boundary can overfit 
        low gamma smother boundary 
    C and gamma are usually tuned together 

SVM are often strong in high dimensions
If you have text classification with huge feature spaces, SVM often performs very well.

#### When does SVM shines?

Very good when:

    Medium-sized datasets
    High-dimensional data
    Text classification
    Bioinformatics
    Complex nonlinear boundaries (with kernels)

Can struggle when:

    Huge datasets (training expensive)
    Massive noise
    Need calibrated probabilities

so in short 

Find boundary
Make margin as wide as possible
Let only critical points define it
Use kernels if straight lines fail







C controls the bias vs variance trade off 
small value of C will have high bias but low variance 