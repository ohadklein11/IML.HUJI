import sys

from IMLearn.utils import split_train_test

sys.path.append("../")
from utils import *
from linear_regression import LinearRegression


def test_linear_regression():
    # uni-variate
    w0, w1 = 1, 2
    x = np.linspace(0, 100, 10)
    y = w1 * x + w0
    fig = go.Figure([go.Scatter(x=x, y=y, name="Real Model", showlegend=True,
                                marker=dict(color="black", opacity=.7),
                                line=dict(color="black", dash="dash", width=1))],
                    layout=go.Layout(title=r"$\text{(1) Simulated Data}$",
                                     xaxis={"title": "x - Explanatory Variable"},
                                     yaxis={"title": "y - Response"},
                                     height=400))
    noiseless_model = LinearRegression()
    fig.show()

    noiseless_model.fit(x.reshape((-1, 1)), y)
    print("noiseless_model Estimated intercept:", noiseless_model.coefs_[0])
    print("noiseless_model Estimated coefficient:", noiseless_model.coefs_[1:])
    print("noiseless loss:", noiseless_model.loss(x, y))

    if "y_" not in locals(): y_ = y
    epsilon = np.random.normal(loc=0, scale=10, size=len(x))
    y = y_ + epsilon
    noisy_model = LinearRegression()
    noisy_model.fit(x.reshape((-1, 1)), y)

    print("noisy_model Estimated intercept:", noisy_model.coefs_[0])
    print("noisy_model Estimated coefficient:", noisy_model.coefs_[1:])
    print("noisy loss:", noisy_model.loss(x, y))

    y_hat = noisy_model.predict(x)

    fig.add_trace(go.Scatter(x=x, y=y, name="Observed Points", mode="markers", line=dict(width=1)))
    fig.update_layout(title=r"$\text{(2) Simulated Data - With Noise}$")
    fig.data = [fig.data[0], fig.data[1]]
    fig.update_layout(title=r"$\text{(3) Fitted Model Over Noisy Data}$")
    fig.add_traces([go.Scatter(x=x, y=y_hat, mode="markers", name="Predicted Responses", marker=dict(color="blue")),
                    go.Scatter(x=x, y=y_hat, mode="lines", name="Fitted Model", line=dict(color="blue", width=1))])
    fig.show()


    # multi-variate
    w0, w1, w2 = 7, 2, 25
    response = lambda x1, x2: w1 * x1 + w2 * x2 + w0

    min_x1, min_x2, max_x1, max_x2 = -10, -10, 10, 10
    xv1, xv2 = np.meshgrid(np.linspace(min_x1, max_x1, 10), np.linspace(min_x2, max_x2, 10))
    surface = response(xv1, xv2)

    x = np.random.uniform((min_x1, min_x2), (max_x1, max_x2), (100, 2))
    y_ = response(x[:, 0], x[:, 1])
    y = y_ + np.random.normal(0, 30, len(x))

    model = LinearRegression()
    model.fit(x, y)
    y_hat = model.predict(x)
    print("Estimated intercept:", model.coefs_[0])
    print("Estimated coefficient:", model.coefs_[1:])
    print("model loss:", model.loss(x, y))

    go.Figure([go.Surface(x=xv1, y=xv2, z=surface, opacity=.5, showscale=False),
               go.Scatter3d(name="Real (noise-less) Points", x=x[:, 0], y=x[:, 1], z=y_, mode="markers",
                            marker=dict(color="black", size=2)),
               go.Scatter3d(name="Observed Points", x=x[:, 0], y=x[:, 1], z=y, mode="markers",
                            marker=dict(color="red", size=2)),
               go.Scatter3d(name="Predicted Points", x=x[:, 0], y=x[:, 1], z=y_hat, mode="markers",
                            marker=dict(color="blue", size=2))],
              layout=go.Layout(
                  title=r"$\text{(4) Bivariate Linear Regression}$",
                  scene=dict(xaxis=dict(title="Feature 1"),
                             yaxis=dict(title="Feature 2"),
                             zaxis=dict(title="Response"),
                             camera=dict(eye=dict(x=-1, y=-2, z=.5)))
              )).show()


def test_house_price():
    import exercises.house_price_prediction as hp
    data, prices = hp.load_data('C:/Users/ohadk/OneDrive/Desktop/Degree/Year 2/IML/IML.HUJI/datasets/house_prices.csv')
    print(data.head(100).to_string())
    data_without_zip = data[data.columns.drop(list(data.filter(regex='zipcode_')))]
    split_train_test(data_without_zip, prices)
    hp.feature_evaluation(data_without_zip, prices)


#test_linear_regression()
test_house_price()
