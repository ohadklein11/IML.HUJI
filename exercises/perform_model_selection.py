from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    eps = np.random.normal(0, np.sqrt(noise), n_samples)
    f = lambda x: (x+3)*(x+2)*(x+1)*(x-1)*(x-2)
    X = np.linspace(-1.2, 2, n_samples)
    y = f(X) + eps

    train_portion = 2/3
    X_train, y_train, X_test, y_test = \
        split_train_test(pd.DataFrame(X), pd.Series(y), train_portion)

    fig = go.Figure([
            go.Scatter(name='true model', x=X, y=f(X), mode='markers+lines', marker_color='green'),
            go.Scatter(name='train set', x=X_train.squeeze(), y=y_train.squeeze(), mode='markers', marker_color='blue'),
            go.Scatter(name='test set', x=X_test.squeeze(), y=y_test.squeeze(), mode='markers', marker_color='red')
    ]) \
        .update_layout(title=f"Data generated from {n_samples} Samples, with noise={noise}",
                       width=1100,
                       xaxis_title="x",
                       yaxis_title="f(x)")
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_errors, validation_errors = [], []
    degrees = [k for k in range(11)]
    for degree in degrees:
        estimator = PolynomialFitting(degree)
        train_error, validation_error = \
            cross_validate(estimator, X_train.squeeze(), y_train.squeeze(), mean_square_error)
        train_errors.append(train_error)
        validation_errors.append(validation_error)

    min_idx = np.argmin(validation_errors)
    selected_degree = degrees[min_idx]
    selected_error = validation_errors[min_idx]

    fig = go.Figure([
            go.Scatter(name='train errors', x=degrees, y=train_errors, mode='markers+lines', marker_color='blue'),
            go.Scatter(name='validation errors', x=degrees, y=validation_errors, mode='markers+lines', marker_color='green'),
            go.Scatter(name='selected degree', x=[selected_degree], y=[selected_error], mode='markers', marker=dict(color='darkred', symbol="x", size=10))
    ]) \
        .update_layout(title=f"Polynomial degree cross-validation errors, with noise={noise}",
                       width=1100,
                       xaxis_title="x",
                       yaxis_title="mean squared error")
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    selected_estimator = PolynomialFitting(selected_degree)
    selected_estimator.fit(X_train.squeeze(), y_train.squeeze())
    test_error = mean_square_error(y_test.squeeze(), selected_estimator.predict(X_test.squeeze()))
    print(f"Sampling {n_samples} samples with noise={noise}.")
    print(f"The value for k* is {selected_degree}.")
    print(f"The validation error(MSE) of k* is {np.round(selected_error, 2)}.")
    print(f"The test error(MSE) of k* is {np.round(test_error, 2)}.\n")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, y_train, X_test, y_test = X[:n_samples], y[:n_samples], X[n_samples:], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas = np.linspace(0, 3, n_evaluations)
    ridge_train_errors, ridge_validation_errors = [], []
    lasso_train_errors, lasso_validation_errors = [], []

    for eval in range(n_evaluations):
        ridge_estimator = RidgeRegression(lambdas[eval])
        lasso_estimator = Lasso(alpha=lambdas[eval], normalize=True, max_iter=10000, tol=1e-4)

        ridge_train_error, ridge_validation_error = \
            cross_validate(ridge_estimator, X_train.squeeze(), y_train.squeeze(), mean_square_error)
        lasso_train_error, lasso_validation_error = \
            cross_validate(lasso_estimator, X_train.squeeze(), y_train.squeeze(), mean_square_error)

        ridge_train_errors.append(ridge_train_error)
        ridge_validation_errors.append(ridge_validation_error)
        lasso_train_errors.append(lasso_train_error)
        lasso_validation_errors.append(lasso_validation_error)

    ridge_min_idx = np.argmin(ridge_validation_errors)
    lasso_min_idx = np.argmin(lasso_validation_errors)
    selected_ridge_lambda = lambdas[ridge_min_idx]
    selected_lasso_lambda = lambdas[lasso_min_idx]
    selected_ridge_error = ridge_validation_errors[ridge_min_idx]
    selected_lasso_error = lasso_validation_errors[lasso_min_idx]

    # ridge graph
    fig = go.Figure([
            go.Scatter(name='train errors', x=lambdas, y=ridge_train_errors, mode='markers', marker_color='blue'),
            go.Scatter(name='validation errors', x=lambdas, y=ridge_validation_errors, mode='markers', marker_color='green'),
            go.Scatter(name='selected lambda', x=[selected_ridge_lambda], y=[selected_ridge_error], mode='markers',
                       marker=dict(color='darkred', symbol="x", size=10))

    ]) \
        .update_layout(title=f"errors of 5-Fold cross validation on Ridge estimator, using different values of lambda.",
                       width=1100,
                       xaxis_title="lambda",
                       yaxis_title="mean squared error")
    fig.show()

    # lasso graph
    fig = go.Figure([
            go.Scatter(name='train errors', x=lambdas, y=lasso_train_errors, mode='markers', marker_color='blue'),
            go.Scatter(name='validation errors', x=lambdas, y=lasso_validation_errors, mode='markers', marker_color='green'),
            go.Scatter(name='selected lambda', x=[selected_lasso_lambda], y=[selected_lasso_error], mode='markers',
                       marker=dict(color='darkred', symbol="x", size=10))
    ]) \
        .update_layout(title=f"errors of 5-Fold cross validation on Lasso estimator, using different values of lambda.",
                       width=1100,
                       xaxis_title="lambda",
                       yaxis_title="mean squared error")
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model

    best_ridge_estimator = RidgeRegression(selected_ridge_lambda)
    best_ridge_estimator.fit(X_train, y_train)

    best_lasso_estimator = Lasso(alpha=selected_lasso_lambda, normalize=True, max_iter=10000, tol=1e-4)
    best_lasso_estimator.fit(X_train, y_train)

    linear_estimator = LinearRegression()
    linear_estimator.fit(X_train, y_train)

    print(f"Best lambda value for Ridge estimator is {selected_ridge_lambda}.")
    print(f"Best lambda value for Lasso estimator is {selected_lasso_lambda}.")
    print(f"MSE of the best Ridge estimator is "
          f"{best_ridge_estimator.loss(X_test, y_test)}.")
    print(f"MSE of the best Lasso estimator is "
          f"{mean_square_error(y_test, best_lasso_estimator.predict(X_test))}.")
    print(f"MSE of Least Squares estimator is "
          f"{linear_estimator.loss(X_test, y_test)}.")


if __name__ == '__main__':
    np.random.seed(0)
 #   select_polynomial_degree()
  #  select_polynomial_degree(noise=0)
   # select_polynomial_degree(1500, 10)
    select_regularization_parameter()
