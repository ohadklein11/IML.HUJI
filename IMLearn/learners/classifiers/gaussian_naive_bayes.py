from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _set_mu_and_var(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Helper func - sets the self.mu_ attribute.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        K, d = self.classes_.size, X.shape[1]
        self.mu_ = [[[] for _ in range(d)] for _ in range(K)]
        self.vars_ = [[[] for _ in range(d)] for _ in range(K)]

        class_indexes = {}
        for k, cls in enumerate(self.classes_):
            class_indexes[cls] = k

        for sample in np.c_[X, y]:
            cls = sample[-1]
            k = class_indexes[cls]
            for j, feature in enumerate(sample[:-1]):
                self.mu_[k][j].append(feature)
                self.vars_[k][j].append(feature)

        for k in range(K):
            for j in range(d):
                self.mu_[k][j] = np.mean(np.array(self.mu_[k][j]))
                self.vars_[k][j] = np.var(np.array(self.vars_[k][j]))

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, class_counts = np.unique(y, return_counts=True)
        K, m, d = self.classes_.size, X.shape[0], X.shape[1]
        self._set_mu_and_var(X, y)
        self.pi_ = np.ndarray(K)
        for k, cls in enumerate(self.classes_):
            self.pi_[k] = class_counts[k] / m

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
        """
        K, m, d = self.classes_.size, X.shape[0], X.shape[1]
        y_pred = np.ndarray(m)
        for i, x in enumerate(X):
            y_cands = np.ndarray(K)
            for k, cls in enumerate(self.classes_):
                y_cands[k] = np.log(self.pi_[k]) - (1 / 2) * \
                             sum([np.log(2 * np.pi * self.vars_[k][j]) +
                                  ((x[j] - self.mu_[k][j]) ** 2) / self.vars_[k][j]
                                  for j in range(d)])
            y_pred[i] = self.classes_[np.argmax(y_cands)]
        return y_pred

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        K, m, d = self.classes_.size, X.shape[0], X.shape[1]

        def likelihood_func(xi, k):
            normal = lambda x, k, j: \
                (1/np.sqrt(2*np.pi*self.vars_[k][j])) * \
                np.exp(-(1/2) * ((xi[j] - self.mu_[k][j])**2)/self.vars_[k][j])
            normal_mult = 1
            for j in range(d):
                normal_mult *= normal(xi, k, j)
            return self.pi_[k] * normal_mult

        likelihoods = np.ndarray((m, K))

        for i, sample in enumerate(X):
            for k, cls in enumerate(self.classes_):
                likelihoods[i][k] = likelihood_func(sample, k)
        return likelihoods

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
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
