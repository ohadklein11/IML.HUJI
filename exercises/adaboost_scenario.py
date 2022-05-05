import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=500, test_size=200):  # todo train 5000 test 500
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train and test errors of AdaBoost in noiseless case
    def default_callback():
        return DecisionStump()

    model = AdaBoost(default_callback, n_learners).fit(train_X, train_y)
    model.fit(train_X, train_y)
    num_learners = np.arange(start=1, stop=n_learners+1)
    train_errors = []
    test_errors = []
    for t in num_learners:
        train_errors.append(model.partial_loss(train_X, train_y, t))
        test_errors.append(model.partial_loss(test_X, test_y, t))
    fig = go.Figure([
            go.Scatter(x=num_learners, y=train_errors,
                       mode='markers + lines', name=r'$Train samples$'),
            go.Scatter(x=num_learners, y=test_errors,
                       mode='markers + lines', name=r'$Test samples^2$')])
    fig.update_layout(title=rf"$\text{{(Q1) Train and test errors of AdaBoost in noiseless case }}$",
                      xaxis=dict(title="Number of learners used"),
                      yaxis=dict(title="loss"))
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    raise NotImplementedError()

    # Question 3: Decision surface of best performing ensemble
    raise NotImplementedError()

    # Question 4: Decision surface with weighted samples
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    # X = np.array([[1, 5],
    #               [1, 7],
    #               [2, 3],
    #               [2, 7],
    #               [3, 7]])
    # y = np.array([1, -1, 1, -1, -1])
    # model = DecisionStump().fit(X, y)
    # print("threshold: ", model.threshold_)
    # print("j: ", model.j_)
    # print("sign: ", model.sign_)
    # print(model.predict(np.array([[1, 3]])))
