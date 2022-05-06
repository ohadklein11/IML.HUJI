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


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train and test errors of AdaBoost
    model = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    model.fit(train_X, train_y)
    num_learners = np.arange(start=1, stop=n_learners + 1)
    train_errors = []
    test_errors = []
    for t in num_learners:
        train_errors.append(model.partial_loss(train_X, train_y, t))
        test_errors.append(model.partial_loss(test_X, test_y, t))
    fig = go.Figure([
        go.Scatter(x=num_learners, y=train_errors,
                   mode='markers + lines', name=r'$Train samples$'),
        go.Scatter(x=num_learners, y=test_errors,
                   mode='markers + lines', name=r'$Test samples$')])
    fig.update_layout(
        title=f"Train and test errors of AdaBoost classifier"
              f"<br>using decision tree with {noise} noise.",
        xaxis=dict(title="Number of learners used"),
        yaxis=dict(title="loss"))
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{i} learners" for i in T],
                        horizontal_spacing=.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(
            lambda X: model.partial_predict(X, t), lims[0], lims[1], showscale=False),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                       mode="markers", showlegend=False,
                       marker=dict(color=(test_y == 1).astype(int),
                                   symbol=class_symbols[test_y.astype(int)],
                                   colorscale=[custom[0], custom[-1]],
                                   line=dict(color="black", width=0.5)))],
            rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.update_layout(
        title=f"Decision Boundaries Of AdaBoost classifier using decision tree"
              f"<br>with {noise} noise, according to the number of learners.",
        width=800, height=800, margin=dict(t=100)
    ).update_xaxes(visible=False).update_yaxes(visible=False)

    fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_size = np.argmin(test_errors) + 1
    from IMLearn.metrics import accuracy
    fig = go.Figure(data=[decision_surface(
        lambda X: model.partial_predict(X, best_size), lims[0], lims[1], showscale=False),
        go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                   mode="markers", showlegend=False,
                   marker=dict(color=(test_y == 1).astype(int),
                               symbol=class_symbols[test_y.astype(int)],
                               colorscale=[custom[0], custom[-1]],
                               line=dict(color="black", width=0.5)))])
    fig.update_layout(
        title=f"Decision surface of best performing ensemble of"
              f"<br>AdaBoost classifier using decision tree with {noise} noise."
              f"<br>Ensemble size={best_size}, Accuracy="
              f"{accuracy(test_y, model.partial_predict(test_X, best_size))}",
        width=800, height=800, margin=dict(t=100)
    ).update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 4: Decision surface with weighted samples
    fig = go.Figure(data=[decision_surface(
        model.predict, lims[0], lims[1], showscale=False),
        go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                   mode="markers", showlegend=False,
                   marker=dict(color=(train_y == 1).astype(int),
                               symbol=class_symbols[train_y.astype(int)],
                               colorscale=[custom[0], custom[-1]],
                               line=dict(color="black", width=1),
                               size=(model.D_ / np.max(model.D_)) * 5))])
    fig.update_layout(
        title=f"Decision surface with weighted samples of AdaBoost"
              f"<br>classifier using decision tree with {noise} noise.",
        width=800, height=800, margin=dict(t=100)
    ).update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
