import numpy as np
from numpy import transpose, linalg
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


TEST_PERCENTAGE = .25


def load_data(filename):
    """
    Load house prices dataset and preprocess data.
    :param filename: Path to house prices dataset
    :return: Design matrix (including intercept) and response vector (prices)
    """
    df = pd.read_csv(filename).dropna().drop_duplicates()
    df["zipcode"] = df["zipcode"].astype(int)

    for c in ["id", "lat", "long", "date"]:
        df = df.drop(c, 1)

    for c in ["price", "sqft_living", "sqft_lot", "sqft_above", "yr_built", "sqft_living15", "sqft_lot15"]:
        df = df[df[c] > 0]
    for c in ["bathrooms", "floors", "sqft_basement", "yr_renovated"]:
        df = df[df[c] >= 0]

    df = df[df["waterfront"].isin([0, 1]) &
            df["view"].isin(range(5)) &
            df["condition"].isin(range(1, 6)) &
            df["grade"].isin(range(1, 15))]

    df["recently_renovated"] = np.where(df["yr_renovated"] >= np.percentile(df.yr_renovated.unique(), 70), 1, 0)
    df = df.drop("yr_renovated", 1)

    df["decade_built"] = (df["yr_built"] / 10).astype(int)
    df = df.drop("yr_built", 1)

    df = pd.get_dummies(df, prefix='zipcode_', columns=['zipcode'])
    df = pd.get_dummies(df, prefix='decade_built_', columns=['decade_built'])

    # Removal of outliers (Notice that there exists methods for better defining outliers
    # but for this course this suffices
    df = df[df["bedrooms"] < 20]
    df = df[df["sqft_lot"] < 1250000]
    df = df[df["sqft_lot15"] < 500000]

    df.insert(0, 'intercept', 1, True)
    return df.drop("price", 1), df.price


def fit_linear_regression(X, y):
    """
    Given a design matrix X and response vector y, fit a linear regression model
    :param X: Design matrix of (m)x(p) for p the number of features and m number of samples
    :param y: Response vector matching given samples
    :return: Coefficients vector of the OLS solution
    """
    return transpose(linalg.pinv(transpose(X))) @ y


def predict(X, w):
    """
    Using a OLS solution, predict the responses for a given set of sample.
    :param X: Matrix of (m)x(p) for p the number of features and m number of samples.
    :param w: OLS coefficients vector of length p.
    :return: The predicted OLS responses for the given samples
    """
    return X @ w


def mse(y, y_pred):
    """
    Calculate the MSE given the true- and prediction- vectors
    :param y: The true response vector
    :param y_pred: The predicted response vector
    :return: MSE of the prediction
    """
    return np.mean((y-y_pred)**2)


def plot_singular_values(X):
    """
    Given a design matrix X, plot the singular values of all non-categorical features
    :param X: The design matrix to use
    """
    X = X.loc[:, ~(X.columns.str.contains('^zipcode_', case=False) |
                   X.columns.str.contains('^decade_built_', case=False))].drop("intercept", 1)
    sv = np.linalg.svd(X, compute_uv=False)
    
    print(pd.DataFrame({"Feature":X.columns, "Singular Value": sv}))
    

    fig = go.Figure(go.Scatter(x=X.columns, y=sv, mode='lines+markers'),
                    layout=go.Layout(title="Scree Plot of Design Matrix Singular Values",
                                     xaxis=dict(title=""), yaxis=dict(title="Singular Values")))
    fig.write_image("singular.values.scree.plot.png")


def feature_evaluation(X, y):
    X = X.loc[:, ~(X.columns.str.contains('^zipcode_', case=False) |
                   X.columns.str.contains('^decade_built_', case=False))].drop("intercept", 1)

    for f in X:
        rho = np.cov(X[f], y)[0, 1] / (np.std(X[f]) * np.std(y))
        
        fig = px.scatter(pd.DataFrame({'x': X[f], 'y': y}), x="x", y="y", trendline="ols",
                         title=f"Correlation Between {f} Values and Response <br>Pearson Correlation {rho}",
                         labels={"x": f"{f} Values", "y": "Response Values"})
        fig.write_image("pearson.correlation.%s.png" % f)


def split_train_test(X, y, test_percentage):
    """
    Randomly split given dataset into training- and testing sets
    :param X: Design matrix to split
    :param y: Response vector to split
    :param test_percentage: Percentage of samples to use as test
    :return: Two tuples of: (train set X, train set y), (test set X, test set y)
    """
    X = X.sample(frac=1)
    y = y.reindex_like(X)

    n = round(test_percentage * len(y))
    return (X[:-n], y[:-n]), (X[-n:], y[-n:])


def _fit_and_test(train, test):
    """
    Fit a linear regression model over a given training set.
    Then evaluate on a given test set
    :param train: Tuple of (samples, response) to use as a training set
    :param test: Tuple of (samples, response) to use as a test set
    :return: MSE over test set of fitted model.
    """
    w = fit_linear_regression(*train)
    return mse(test[1], predict(test[0], w))


if __name__ == "__main__":
    X, y = load_data('..\\datasets\\house_prices.csv')

    #plot_singular_values(X)
    feature_evaluation(X, y)

    train, test = split_train_test(X, y, TEST_PERCENTAGE)
    results = []
    for i in range(1, 101):
        n = max(round(len(train[1]) * (i/100)), 1)
        results.append(_fit_and_test((train[0][:n], train[1][:n]), test))

    fig = go.Figure(go.Scatter(x=list(range(1, len(results)+1)), y=results, mode="markers"),
                    layout=go.Layout(title="Model Evaluation Over Increasing Portions Of Training Set",
                                     xaxis=dict(title="Percentage of Training Set"),
                                     yaxis=dict(title="MSE Over Test Set")))
    fig.write_image("mse.over.training.percentage.png")