from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"

def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    num_draws = 1000
    mean, var = 10, 1
    X = np.random.normal(mean, var, num_draws)
    uni_gaussian = UnivariateGaussian()
    uni_gaussian.fit(X)
    print((uni_gaussian.mu_, uni_gaussian.var_))

    # Question 2 - Empirically showing sample mean is consistent
    num_models = 100
    distances = np.ndarray((num_models,))
    sample_sizes = np.arange(10, 1001, 10)
    model = UnivariateGaussian()
    for i in range(num_models):
        model.fit(X[:int(sample_sizes[i])])
        distances[i] = np.abs(model.mu_ - mean)

    data = (sample_sizes, distances)
    fig = go.Figure(data=go.Scatter(x=data[0], y=data[1]))
    fig.update_layout(
        title="Distance between normal distribution's mean estimation and "
              "true value by Sample size",
        xaxis_title="Distance between mean estimation and real value",
        yaxis_title="Sample size"
    )
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = uni_gaussian.pdf(X)
    data = (X, pdfs)
    fig2 = go.Figure(data=go.Scatter(x=data[0], y=data[1], mode='markers'))
    fig2.update_layout(
        title="PDF by Sample size",
        xaxis_title="Sample size",
        yaxis_title="PDF"
    )
    fig2.show()
    # I expect to see the gaussian bell since the real density function f will
    # take each sample x to f(x), so the PDF which estimates the distribution
    # should take each x to a very close value to f(x).


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
