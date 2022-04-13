from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    # removing irrelevant data
    df.drop(['id', 'date', 'lat', 'long'], axis=1, inplace=True)

    # removing samples with invalid values
    for col_name in ('price', 'sqft_living', 'floors',
                     'sqft_above', 'yr_built', 'zipcode',
                     'sqft_living15', 'sqft_lot15'):
        df.drop(df[df[col_name] <= 0].index, inplace=True)

    for col_name in ('bedrooms', 'bathrooms', 'sqft_lot',
                     'sqft_basement', 'yr_renovated'):
        df.drop(df[df[col_name] < 0].index, inplace=True)
    # removing samples with null values
    df.dropna(axis=0, inplace=True)

    # making sure categorical columns are in correct range
    waterfront_min, waterfront_max = 0, 1
    df.drop(df[(df['waterfront'] < waterfront_min)
               | (df['waterfront'] > waterfront_max)].index, inplace=True)
    view_min, view_max = 0, 4
    df.drop(df[(df['view'] < view_min)
               | (df['view'] > view_max)].index, inplace=True)
    condition_min, condition_max = 1, 5
    df.drop(df[(df['condition'] < condition_min)
               | (df['condition'] > condition_max)].index, inplace=True)
    grade_min, grade_max = 1, 13
    df.drop(df[(df['grade'] < grade_min)
               | (df['grade'] > grade_max)].index, inplace=True)

    # add additional columns
    yr_built_range = max(df['yr_built']) - min(df['yr_built'])
    # add newly_built as top 10% of yr_built
    yr_built_lower_bound = max(df['yr_built']) - yr_built_range / 10
    df['newly_built'] = \
        np.where((df['yr_built'] > yr_built_lower_bound)
                 | (df['yr_renovated'] > yr_built_lower_bound), 1, 0)

    # dummy values for relevant columns - using one-hot encoding
    df['zipcode'] = df['zipcode'].astype(int)
    df = pd.get_dummies(df, prefix='zipcode', columns=['zipcode'])

    # add intercept
    df.insert(0, 'intercept', 1, True)

    return df.drop('price', axis=1), df['price']


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X:
        # note that the covariance between X and y is stored in index (0,1)
        # (or, simetrically, (1,0)) of the covariance matrix,
        # since (0,0) is the variance of X and (1,1) is the variance of y.
        cov_X_y = np.cov(X[feature], y)[0][1]
        dev_X, dev_y = np.std(X[feature]), np.std(y)
        pearson_correlation_X_y = cov_X_y / (dev_X * dev_y)

        response = y.name if y.name else 'Response'
        fig = px.scatter(pd.DataFrame({'feature': X[feature], 'response': y}),
                         x='feature', y='response',
                         title=f'Correlation between {feature} and {response} '
                               f'in house prices data,<br>with Pearson '
                               f'Correlation of {pearson_correlation_X_y}',
                         labels={'feature': feature, 'response': response})
        fig.write_image(f"{output_path}\\{feature}_scatter.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    data, prices = load_data('..\\datasets\\house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    # note that the current data holds lots of features with zipcode_
    # for which it is irrelevant to draw scatter plots.
    data_without_zip = \
        data[data.columns.drop(list(data.filter(regex='zipcode_')))]
    data_without_zip_and_intercept = \
        data_without_zip[data_without_zip.columns.drop('intercept')]
    feature_evaluation(data_without_zip_and_intercept, prices)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(data, prices, .75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    n_training, n_tests = train_X.shape[0], test_X.shape[0]
    percentages, means, confidence_intervals_minus, confidence_intervals_plus = [], [], [], []
    model = LinearRegression()
    for p in range(10, 101):
        losses = []
        n_samples = int(np.ceil(n_training * p / 100))
        for _ in range(10):
            data_samples = train_X.sample(n_samples, axis=0)
            prices_samples = prices[data_samples.index]
            model.fit(np.array(data_samples), prices_samples)
            loss = model.loss(np.array(test_X), np.array(test_y))
            losses.append(loss)
        mean_loss, dev_loss = \
            np.mean(losses, axis=0), np.std(losses, axis=0)
        percentages.append(p)
        means.append(mean_loss)
        confidence_intervals_minus.append(mean_loss - 2 * dev_loss)
        confidence_intervals_plus.append(mean_loss + 2 * dev_loss)
    fig = go.Figure([go.Scatter(x=percentages, y=means, mode="markers+lines",
                                name="Mean Prediction", line=dict(dash="dash"),
                                marker=dict(color="green", opacity=.7)),
                    go.Scatter(x=percentages,
                               y=confidence_intervals_minus,
                               fill=None, mode="lines",
                               line=dict(color="lightgrey"), showlegend=False),
                    go.Scatter(x=percentages,
                               y=confidence_intervals_plus,
                               fill='tonexty', mode="lines",
                               line=dict(color="lightgrey"), showlegend=False)])
    fig.update_layout(
        title="(Q4) Loss over test set of Linear Regression house price model,"
              " by samples percent from training set.",
        xaxis_title="samples percent from training set",
        yaxis_title="loss over test set"
    )
    fig.show()
