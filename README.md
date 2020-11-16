# Gaussian-Mixture-Model-Classifier
#### The goal of this assignment is to classify using Gaussian Mixture Model (GMM) classifier.

## Main description

In this homework, the data sets of two artificial classification problems will be used. 

Each data set is
for a 2-class classification problem and consists of N training data and N test data, where N=225 for
problem 1 and N=220 for problem 2. The input data are within a range of [-1,1]^2, and the target
value is either 0 or 1, meaning class 1 or class 2, respectively. Four data files are provided for each
problem as follows (where ? is equal to 1 or 2):

    • p?_train_input.txt: training samples (Nx2 matrix)
    • p?_train_target.txt: target category (0 or 1) for each training sample (Nx1 vector)
    • p?_test_input.txt: test samples (Nx2 matrix)
    • p?_test_target.txt: target category for each test sample (Nx1 vector)

Write your code for training and testing a GMM classifier. Try different structures of GMMs, 
including the number of Gaussian components and the type of covariance matrices (spherical,
diagonal, and full), and observe how the performance changes.

Your objective is not to simply write code to train and test GMMs, but to develop your own research
questions and conduct experiments to answer the questions. Report your results with thorough
discussion. Some example questions are (but not limited to):

    • How does the choice of the number of Gaussian components influence the performance?
    • Which is better among different types of covariance matrices? In which sense?
    • Does the performance vary according to the initialization of the model parameters?
    • What is a good strategy to initialize the model parameters?
    • What is a good strategy to determine the convergence of the EM algorithm?
    • Does overfitting occur?

Some tips are:

    • The primary measure for performance will be the test accuracy. However, the training time
    and run time can be also considered.
    • Since the input data are 2-dimensional, you can easily visualize the given data and results,
    such as the decision boundary, the final locations and shapes of the Gaussian components
    after training, the process that the Gaussian components ‘move’ in the input space during
    training, etc.

## Requirements

Ubuntu 16.04
CUDA 10.1
cuDNN 7.5
Python 3.6
sklearn 0.15.0.
numpy 1.15.4
matplotlib 2.1.0

## Testing

```bash
# visualization process gmm
python3 main_R.m

# original 2-class classifciation
python3 main_R_0.m

# for check number of component
python3 main_R_1.m

# for visualization decision boundary
python3 main_R_2.m