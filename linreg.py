import numpy as np
from sklearn.linear_model import LinearRegression

def closed_form(data, target, intercept=True):
    """This computes the linear regression parameters by it's direct closed formula, the normal equation.

    Keyword Arguments:
    data(np array) --your data in tabular format
    target(np array) --your target column
    intercept(boolean) --Do you wish to include Intercept

    returns a parameter list in np format
    """
    X = data  # data represented as a 2D matrix
    X_b = np.c_[np.ones((data.shape[0],1)), X]  # This adds the vector of ones to represent the intercept
    X_T = X_b.T  # your data transposed

    inv_dp = np.linalg.inv(X_T.dot(X_b))
    best_theta = inv_dp.dot(X_T).dot(y)  # this will give you your best parameters

    return best_theta

def gradient_descent(data, target):
    """This computes the linear regression parameters by using Gradient Descent on the MSE cost functionself.

    Keyword Arguments:
    data(np array) --your data in tabular format
    target(np array) --your target column

    returns a parameter list in np format
    """
    
