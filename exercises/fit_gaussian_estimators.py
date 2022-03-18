from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"

def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    print("(Q1)")
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
        title="(Q2) Distance between normal distribution's mean estimation "
              "and true value by Sample size",
        xaxis_title="Distance between mean estimation and real value",
        yaxis_title="Sample size"
    )
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = uni_gaussian.pdf(X)
    data = (X, pdfs)
    fig2 = go.Figure(data=go.Scatter(x=data[0], y=data[1], mode='markers'))
    fig2.update_layout(
        title="(Q3) PDF by Sample size",
        xaxis_title="Sample size",
        yaxis_title="PDF"
    )
    fig2.show()
    # I expect to see the gaussian bell since the real density function f will
    # take each sample x to f(x), so the PDF which estimates the distribution
    # should take each x to a very close value to f(x).


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    print("(Q4)")
    num_draws = 1000
    mean = np.array([0, 0, 4, 0])
    cov = np.array([[1,   0.2, 0,   0.5],
                    [0.2, 2,   0,   0],
                    [0,   0,   1,   0],
                    [0.5, 0,   0,   1]])

    X = np.random.multivariate_normal(mean, cov, num_draws)
    multi_gaussian = MultivariateGaussian()
    multi_gaussian.fit(X)
    print("expectation:", multi_gaussian.mu_,
          "covariance matrix:", multi_gaussian.cov_,
          sep='\n')

    # Question 5 - Likelihood evaluation
    num_draws = 200
    space_edge = 10
    f1 = np.linspace(-space_edge, space_edge, num_draws)
    f3 = np.linspace(-space_edge, space_edge, num_draws)
    log_likelihoods = np.array([multi_gaussian.log_likelihood(
            np.array([f1[i], 0, f3[j], 0]), cov, X)
        for i in range(num_draws) for j in range(num_draws)]).reshape((num_draws, num_draws))

    fig = go.Figure(go.Heatmap(x=f1, y=f3, z=log_likelihoods))
    fig.update_layout(
        title="(Q5) log likelihood derived from mean [f1, 0, f3, 0] and "
              "given covariance matrix",
        xaxis_title="f1",
        yaxis_title="f3"
    )
    fig.show()
    # I am able to learn from the plot that the closer to the true mean
    # [0, 0, 4, 0] the more accurate the estimation is.

    # Question 6 - Maximum likelihood
    print("(Q6)")
    max_idx = np.unravel_index(
        np.argmax(log_likelihoods), log_likelihoods.shape)
    print("max likelihood:", (np.round(f1[max_idx[0]], 3),
                              np.round(f3[max_idx[1]], 3)))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
