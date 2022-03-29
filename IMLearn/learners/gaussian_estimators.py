from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        m = X.shape[0]

        def sample_mean_estimator(X: np.ndarray):
            return np.sum(X) / m

        def sample_variance_estimator(X: np.ndarray):
            sum_elems = np.array(
                [np.power(X[i] - self.mu_, 2) for i in range(m)])
            divisor = m if self.biased_ else m + 1
            return (np.sum(sum_elems)) / divisor

        self.mu_ = sample_mean_estimator(X)
        self.var_ = sample_variance_estimator(X)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        m = X.shape[0]

        def formula(x):
            exp = np.exp(-(np.power(x - self.mu_, 2) / (2 * self.var_)))
            return exp / (np.sqrt(2 * np.pi * self.var_))

        return np.array([formula(X[i]) for i in range(m)])

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        m = X.shape[0]
        exp_sum = np.sum([np.power(X[i] - mu, 2) for i in range(m)])
        # following formula calculated in class
        return -(m / 2) * np.log(2 * np.pi) \
               - (m / 2) * np.log(sigma) - (exp_sum / (2 * sigma))


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        m = X.shape[0]

        def sample_mean_estimator(X: np.ndarray):
            return (np.sum(X, axis=0)) / m

        def sample_covariance_estimator(X: np.ndarray):
            # first subtract the empirical mean from each sample
            X_centered = np.array([X[i] - self.mu_ for i in range(m)])
            # then follow the definition of sample covariance matrix
            return np.dot(X_centered.T, X_centered) / (m - 1)

        self.mu_ = sample_mean_estimator(X)
        self.cov_ = sample_covariance_estimator(X)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        m = X.shape[0]
        d = X.shape[1]

        def formula(x: np.ndarray):
            # following formula from course book - definition 1.2.5
            exp = np.exp(
                -(1 / 2) * np.dot(
                    (x - self.mu_).T,
                    np.dot(np.linalg.inv(self.cov_), (x - self.mu_))))
            divisor = np.sqrt(
                np.power((2 * np.pi), d) * np.linalg.det(self.cov_))
            return exp / divisor

        return np.array([formula(X[i]) for i in range(m)])

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """

        m, d = X.shape[0], X.shape[1]
        # following formula calculated in Ex01 Q09
        exp_sum = np.sum((X - mu) @ np.linalg.inv(cov) * (X - mu))
        return -(m * d / 2) * np.log(2 * np.pi) \
               - (m / 2) * np.log(np.linalg.det(cov)) \
               - (exp_sum / 2)
