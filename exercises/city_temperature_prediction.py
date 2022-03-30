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
    monthly_temps = []
    for _ in range(12):
        monthly_temps.append([])
    for year in israeli_data['Year'].unique():
        temps_per_day = []
        for day in range(1, 366):
            temp_row = israeli_data[(israeli_data['Year'] == year) &
                                    (israeli_data['DayOfYear'] == day)]
            if temp_row.Temp.size == 1:
                temp = temp_row.Temp.values[0]
                temps_per_day.append((day, temp))
                month_idx = temp_row.Month.values[0] - 1
                monthly_temps[month_idx].append(temp)
        temps_per_day = pd.DataFrame(temps_per_day)
        scatters.append(go.Scatter(x=temps_per_day[0],
                                   y=temps_per_day[1],
                                   mode='markers', name=f'{year}'))
    fig = go.Figure(data=scatters)
    fig.update_layout(
        title="(Q3.2.2) Temperature in Israel by day of year",
        xaxis_title="Day of Year",
        yaxis_title="Temperature")
    # fig.show()
    # it seems like a polynomial of degree 3 might be fitting the graph

    # plot the standard deviation of the daily temperatures for each month

    monthly_devs = [np.std(temps) for temps in monthly_temps]
    fig = go.Figure(data=[go.Bar(x=np.linspace(1, 12, 12).astype(int),
                                 y=monthly_devs)])
    fig.update_layout(
        title="(Q3.2.2) Monthly deviation of temperatures",
        xaxis_title="Month",
        yaxis_title="Temperature Deviation")
    fig.show()

    # Question 3 - Exploring differences between countries
    raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()
