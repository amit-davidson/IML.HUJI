from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    X_folds = np.array_split(X, cv, axis=1)
    y_folds = np.array_split(X, cv, axis=1)

    train_score = 0
    validation_score = 0
    for i in range(cv):
        train_set_X = np.concatenate(X_folds[:i] + X_folds[i + 1:], axis=0)
        train_set_y = np.concatenate(y_folds[:i] + y_folds[i + 1:], axis=0)
        estimator.fit(train_set_X, train_set_y)
        train_score += scoring(y_folds[i], estimator.predict(X_folds[i]))
        validation_score += scoring(y_folds[i], estimator.predict(X_folds[i]))
    return train_score / cv, validation_score / cv
