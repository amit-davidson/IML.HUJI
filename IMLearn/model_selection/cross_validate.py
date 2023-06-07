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
    idx = np.arange(len(X))
    folds = np.array_split(idx, cv)

    train_score = 0
    validation_score = 0
    for i in range(cv):
        train_msk = get_mask(i, folds, len(X))
        train_set_X = X[train_msk]
        train_set_y = y[train_msk]
        copied = deepcopy(estimator)
        copied.fit(train_set_X, train_set_y)
        train_score += scoring(train_set_y, copied.predict(train_set_X))
        validation_score += scoring(y[folds[i]], copied.predict(X[folds[i]]))

    return train_score / cv, validation_score / cv

def get_mask(idx, folds, size):
    mask = np.full((size,), True)
    for i in range(size):
        if i in folds[idx]:
            mask[i] = False
    return mask
