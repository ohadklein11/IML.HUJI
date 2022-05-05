from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product

from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # todo error is not here
        m, d = X.shape[0], X.shape[1]
        min_err = 1
        for val, feature in product([1, -1], range(d)):
            threshold, threshold_err = \
                self._find_threshold(X[:, feature], y, val)
            if threshold_err < min_err:
                self.threshold_ = threshold
                self.j_ = feature
                self.sign_ = val
                min_err = threshold_err

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        m = X.shape[0]
        return np.array(
            [self.sign_ if X[i][self.j_] >= self.threshold_ else -self.sign_
             for i in range(m)])

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # todo error is not here.
        thr, thr_err = 0, 1
        m = values.shape[0]
        values_indexes = values.argsort(axis=0)
        values = values[values_indexes]
        labels = labels[values_indexes]
        i, prev_i = 0, 0
        pred_vals = np.array([sign] * m)
        while i < m:
            err = sum(np.abs(labels[pred_vals != np.sign(labels)])) / m
            if err < thr_err:
                thr = (values[i] + values[prev_i]) / 2
                thr_err = err
            # inc i to next value
            prev_i = i
            pred_vals[i] = -sign
            i += 1
            while i < m and values[i] == values[i-1]:
                # avoid multiple checks for same value
                pred_vals[i] = -sign
                i += 1
        return thr, thr_err

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under misclassification loss function
        """
        return misclassification_error(y, self._predict(X))
