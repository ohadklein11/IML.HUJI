import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=True)

    # removing samples with invalid values
    df.drop(df[df['Temp'] <= -12].index, inplace=True)

    # add additional columns
    df['DayOfYear'] = pd.to_datetime(df['Date']).dt.dayofyear

    # add intercept
    df.insert(0, 'intercept', 1, True)

    return df


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data('..\\datasets\\City_Temperature.csv')

    # Question 2 - Exploring data for specific country

    israeli_data = data[data['Country'] == 'Israel']
    scatters = []
    for year in israeli_data['Year'].unique():
        temps_per_day = []
        for day in range(1, 366):
            temp_row = israeli_data[(israeli_data['Year'] == year) &
                                    (israeli_data['DayOfYear'] == day)]
            if temp_row.Temp.size == 1:
                temp = temp_row.Temp.values[0]
                temps_per_day.append((day, temp))
        temps_per_day = pd.DataFrame(temps_per_day)
        scatters.append(go.Scatter(x=temps_per_day[0],
                                   y=temps_per_day[1],
                                   mode='markers', name=f'{year}'))
    fig = go.Figure(data=scatters)
    fig.update_layout(
        title="(Q2) Temperature in Israel by day of year",
        xaxis_title="Day of Year",
        yaxis_title="Temperature")
    fig.show()
    # it seems like a polynomial of degree 3 might be fitting the graph

    # plot the standard deviation of the daily temperatures for each month
    monthly_devs = israeli_data.groupby('Month').Temp.agg([np.std])
    fig = go.Figure(data=[go.Bar(x=np.linspace(1, 12, 12).astype(int),
                                 y=monthly_devs['std'])])
    fig.update_layout(
        title="(Q2) Monthly deviation of temperatures",
        xaxis_title="Month",
        yaxis_title="Temperature Deviation")

    fig.show()

    # Question 3 - Exploring differences between countries
    by_month_and_country = data.groupby(['Country', 'Month']).Temp.agg([np.mean, np.std])
    fig = px.line(by_month_and_country,
                  x=by_month_and_country.index.get_level_values('Month'),
                  y='mean', error_y='std',
                  color=by_month_and_country.index.get_level_values('Country'))
    fig.update_layout(
        title="(Q3) Temperature mean +- deviation of Countries by Months",
        xaxis_title="Month",
        yaxis_title="Temperature mean +- deviation")
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_Y, test_X, test_Y = \
        split_train_test(israeli_data.drop('Temp', axis=1), israeli_data['Temp'])
    print('Test error recorded for each value of k in range [1,10]:')
    losses = []
    min_loss = None
    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(train_X['DayOfYear'], train_Y)
        loss = round(model.loss(test_X['DayOfYear'], test_Y), 2)
        losses.append(loss)
        min_loss = (k, loss) if min_loss is None or loss < min_loss[1] else min_loss
        print(f'loss for polynomial fitting of degree {k} is: {losses[k - 1]}')
    fig = go.Figure(data=[go.Bar(x=np.linspace(1, 10, 10).astype(int),
                                 y=losses)])
    fig.update_layout(
        title="(Q4) Losses of polynomial fitting of degree k "
              "over Temperature prediction in israel",
        xaxis_title="k",
        yaxis_title="loss of Polynomial fitting of degree k")
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    # selecting k with smallest loss
    k = min_loss[0]
    train_X, train_Y = israeli_data.drop('Temp', axis=1), israeli_data['Temp']
    test_data = data[data['Country'] != 'Israel']
    model = PolynomialFitting(k).fit(train_X['DayOfYear'], train_Y)
    country_losses = []
    for country in test_data['Country'].unique():
        country_test = test_data[test_data['Country'] == country]
        country_test_X = country_test.drop('Temp', axis=1)
        country_test_Y = country_test['Temp']
        loss = round(model.loss(country_test_X['DayOfYear'], country_test_Y), 2)
        country_losses.append(loss)
    fig = go.Figure(data=[go.Bar(x=test_data['Country'].unique(),
                                 y=country_losses)])
    fig.update_layout(
        title=f"(Q5) Losses of different countries in polynomial fitting "
              f"of degree {k} <br>over Temperature prediction in israel",
        xaxis_title="country",
        yaxis_title="loss of Polynomial fitting of degree k in israel")
    fig.show()
