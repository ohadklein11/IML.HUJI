from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `LDA.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _set_mu(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
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
        # set for each class its
        class_indexes = {}
        for k, cls in enumerate(self.classes_):
            class_indexes[cls] = k

        for sample in np.c_[X, y]:
            cls = sample[-1]
            k = class_indexes[cls]
            for j, feature in enumerate(sample[:-1]):
                self.mu_[k][j].append(feature)

        for k in range(K):
            for j in range(d):
                self.mu_[k][j] = np.mean(np.array(self.mu_[k][j]))

    def _set_cov(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Helper func - sets the self.cov_ attribute.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        K, m, d = self.classes_.size, X.shape[0], X.shape[1]

        class_indexes = {}
        for k, cls in enumerate(self.classes_):
            class_indexes[cls] = k

        self.cov_ = (1 / m) * np.sum(
            [np.outer((X[i] - self.mu_[class_indexes[y[i]]]),
                      (X[i] - self.mu_[class_indexes[y[i]]]))
             for i in range(m)], axis=0)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, class_counts = np.unique(y, return_counts=True)
        K, m, d = self.classes_.size, X.shape[0], X.shape[1]
        self._set_mu(X, y)
        self._set_cov(X, y)
        self._cov_inv = np.linalg.inv(self.cov_)
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
        K, m = self.classes_.size, X.shape[0]
        y_cands = np.ndarray((K, m))
        for k, cls in enumerate(self.classes_):
            ak = self._cov_inv @ self.mu_[k]
            bk = np.log(self.pi_[k]) - (1 / 2) * float(self.mu_[k] @ ak)
            y_cands[k] = ak @ X.T + bk
        return self.classes_[np.argmax(y_cands, axis=0)]

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

        def likelihood_func(i, k):
            xi_mu_yi = X[i] - self.mu_[k]
            normal = (1 / np.sqrt(((2 * np.pi) ** d) * np.linalg.det(self.cov_))) * \
                     np.exp(-(1 / 2) * xi_mu_yi.T @ self._cov_inv @ xi_mu_yi)
            return normal * self.pi_[k]

        K, m, d = self.classes_.size, X.shape[0], X.shape[1]
        likelihoods = np.ndarray((m, K))

        for i, sample in enumerate(X):
            for k, cls in enumerate(self.classes_):
                likelihoods[i][k] = likelihood_func(i, k)
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
