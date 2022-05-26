from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
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
    train_scores, validation_scores = [], []
    X_folds = np.array_split(X, cv)
    y_folds = np.array_split(y, cv)
    for i in range(cv):
        X_train = np.concatenate(X_folds[:i] + X_folds[i+1:])
        y_train = np.concatenate(y_folds[:i] + y_folds[i+1:])
        estimator.fit(X_train, y_train)
        X_validation, y_validation = X_folds[i], y_folds[i]
        train_scoring = scoring(y_train, estimator.predict(X_train))
        validation_scoring = scoring(y_validation, estimator.predict(X_validation))
        train_scores.append(train_scoring)
        validation_scores.append(validation_scoring)

    train_score = np.mean(train_scores)
    validation_score = np.mean(validation_scores)
    return train_score, validation_score




