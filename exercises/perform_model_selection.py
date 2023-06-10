from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
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
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
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

    cv: int = 5
    # Question 1 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)

    idx = np.arange(len(X))
    chosen_idx = np.random.choice(idx, n_samples)
    mask = np.zeros(len(X), dtype=bool)
    mask[chosen_idx] = True

    train_X, train_y, test_X, test_y = X[mask], y[mask], X[~mask], y[~mask]

    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    train_scores_r = np.empty(shape=(n_evaluations,))
    validation_scores_r = np.empty(shape=(n_evaluations,))
    train_scores_l = np.empty(shape=(n_evaluations,))
    validation_scores_l = np.empty(shape=(n_evaluations,))

    ridge_alphas = np.linspace(0.005, 1, num=n_evaluations)
    lasso_alphas = np.linspace(.1, 1.3, num=n_evaluations)
    for i in range(0, n_evaluations):
        train_score_r, validation_score_r = cross_validate(
            RidgeRegression(ridge_alphas[i], True), train_X, train_y,
            mean_square_error, cv)
        train_score_l, validation_score_l = cross_validate(
            Lasso(alpha=lasso_alphas[i]), train_X, train_y, mean_square_error,
            cv)

        train_scores_r[i] = train_score_r
        validation_scores_r[i] = validation_score_r
        train_scores_l[i] = train_score_l
        validation_scores_l[i] = validation_score_l

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[f"Ridge", "Lasso"],
                        horizontal_spacing=0.1, vertical_spacing=0.07,
                        shared_xaxes=True) \
        .update_xaxes(title="Regularization parameter")

    fig.add_trace(go.Scatter(x=ridge_alphas, y=train_scores_r,
                             mode='lines', name="Train Error Ridge"), row=1,
                  col=1)
    fig.add_trace(
        go.Scatter(x=ridge_alphas, y=validation_scores_r,
                   mode='lines', name="Validation Error Ridge"), row=1, col=1)
    fig.add_trace(go.Scatter(x=lasso_alphas, y=train_scores_l,
                             mode='lines', name="Train Error Lasso"), row=1,
                  col=2)

    fig.add_trace(go.Scatter(x=lasso_alphas,
                             y=validation_scores_l, mode='lines',
                             name="Validation Error Lasso"), row=1, col=2)

    fig.update_layout(
        title=f"Train and Validation Errors of Lasso and Ridge as a function "
              f"of the Regularization parameter",
        yaxis_title="Error",
        title_x=0.5,
    )
    fig.write_image(f"CrossValidationModelSelection.png")

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    best_r_i = np.argmin(validation_scores_r)
    print("best_ridge: ", ridge_alphas[best_r_i])
    best_l_i = np.argmin(validation_scores_l)
    print("best_lasso: ", lasso_alphas[best_l_i])

    print(f"Least Squares: ",
          LinearRegression().fit(train_X, train_y).loss(test_X, test_y))
    print(f"Ridge: ", RidgeRegression(lam=ridge_alphas[best_r_i]).fit(train_X,
                                                                      train_y).loss(
        test_X, test_y))
    print(f"Lasso: ", mean_square_error(test_y, Lasso(
        alpha=lasso_alphas[best_l_i]).fit(train_X, train_y).predict(test_X)))


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
