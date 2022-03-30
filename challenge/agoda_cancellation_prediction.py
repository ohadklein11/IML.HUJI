from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
from IMLearn.base import BaseEstimator

import numpy as np
import pandas as pd


def _parse_cancellation_policy_code(data: pd.DataFrame):
    interval = 20
    for value in range(0, 366, interval):
        data[f'{value}-{value + interval}_N'] = 0
        data[f'{value}-{value + interval}_P'] = 0
    data['no_show_N'] = 0
    data['no_show_P'] = 0
    for index, row in data.iterrows():
        value = row['cancellation_policy_code']
        values = str(value).split(sep='_')
        for val in values:
            if 'D' in val:
                days, price = val.split(sep='D')
                days = min(int(days), 365)  # no more than a year in consideration
                days_interval = (int(days) // 20) * 20
                label_N = f'{days_interval}-{days_interval + interval}_N'
                label_P = f'{days_interval}-{days_interval + interval}_P'
            else:
                price = val
                label_N = 'no_show_N'
                label_P = 'no_show_P'
            if not price[:-1].isnumeric():
                continue
            if price[-1] == 'N':
                data.at[index, label_N] = int(price[:-1])
            elif price[-1] == 'P':
                data.at[index, label_P] = int(price[:-1])


def load_data(filename: str, test_filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    test_filename: str
        Path to test dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    full_data = pd.read_csv(filename).drop_duplicates()
    test_data = pd.read_csv(test_filename)
    full_data = full_data.append(test_data).reset_index()
    test_size = test_data.shape[0]

    # take care of dates
    full_data[['booking_datetime', 'checkin_date', 'checkout_date']] = \
        full_data[['booking_datetime', 'checkin_date', 'checkout_date']]. \
            apply(pd.to_datetime)
    full_data['checkin_month'] = pd.to_datetime(full_data['checkin_date']).dt.month
    full_data['checkin_weekday'] = pd.to_datetime(full_data['checkin_date']).dt.dayofweek
    full_data['days_stay'] = \
        (full_data['checkout_date'] - full_data['checkin_date']).dt.days
    full_data['days_until_checkin'] = \
        (full_data['checkin_date'] - full_data['booking_datetime']).dt.days

    # adding new columns
    full_data['live_where_ordered'] = \
        np.where(full_data['hotel_country_code'] == full_data['origin_country_code'], 1, 0)

    # todo reconsider adding language, currency

    # filling null values of relevant columns
    for request in ['request_nonesmoke', 'request_latecheckin',
                    'request_highfloor', 'request_largebed',
                    'request_twinbeds', 'request_airport',
                    'request_earlycheckin']:
        full_data[request].fillna(np.mean(full_data[request]), inplace=True)

    # dummy values for relevant columns - using one-hot encoding
    full_data = pd.get_dummies(full_data, prefix='accommadation_type_name', columns=['accommadation_type_name'])
    full_data = pd.get_dummies(full_data, prefix='hotel_country_code', columns=['hotel_country_code'])
    full_data = pd.get_dummies(full_data, prefix='origin_country_code', columns=['origin_country_code'])
    full_data = pd.get_dummies(full_data, prefix='is_first_booking', columns=['is_first_booking'])
    full_data = pd.get_dummies(full_data, prefix='charge_option', columns=['charge_option'])

    # dummy values with threshold
    s = full_data['hotel_area_code'].value_counts()
    sdumm = pd.get_dummies(
        full_data.loc[full_data['hotel_area_code'].isin(s.index[s >= 50]),
                      'hotel_area_code'], prefix='hotel_area_code')
    full_data = pd.concat([full_data, sdumm.reindex(full_data.index).fillna(0)], axis=1)

    s = full_data['hotel_brand_code'].value_counts()
    sdumm = pd.get_dummies(
        full_data.loc[full_data['hotel_brand_code'].isin(s.index[s >= 50]),
                      'hotel_brand_code'], prefix='hotel_brand_code')
    full_data = pd.concat([full_data, sdumm.reindex(full_data.index).fillna(0)], axis=1)

    s = full_data['hotel_chain_code'].value_counts()
    sdumm = pd.get_dummies(
        full_data.loc[full_data['hotel_chain_code'].isin(s.index[s >= 50]),
                      'hotel_chain_code'], prefix='hotel_chain_code')
    full_data = pd.concat([full_data, sdumm.reindex(full_data.index).fillna(0)], axis=1)

    s = full_data['hotel_city_code'].value_counts()
    sdumm = pd.get_dummies(
        full_data.loc[full_data['hotel_city_code'].isin(s.index[s >= 50]),
                      'hotel_city_code'], prefix='hotel_city_code')
    full_data = pd.concat([full_data, sdumm.reindex(full_data.index).fillna(0)], axis=1)

    # take care of cancellation policy
    _parse_cancellation_policy_code(full_data)

    # removing irrelevant data
    full_data.drop(['h_booking_id', 'hotel_id', 'hotel_live_date', 'h_customer_id',
                    'customer_nationality', 'guest_nationality_country_name',
                    'language', 'original_payment_method', 'original_payment_type',
                    'original_payment_currency', 'is_user_logged_in',
                    'cancellation_policy_code', 'booking_datetime',
                    'checkin_date', 'checkout_date', 'hotel_area_code',
                    'hotel_city_code', 'hotel_brand_code', 'hotel_chain_code'],
                   axis=1, inplace=True)

    if 'cancellation_datetime' in full_data:
        data, results = full_data.drop('cancellation_datetime', axis=1), \
                        full_data['cancellation_datetime']
        data = data.dropna()

        # take care of results
        results.loc[~results.isnull()] = 1
        results.loc[results.isnull()] = 0

        return data.head(-test_size), results.head(-test_size).astype(int), \
               data.tail(test_size)


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    train_X, train_y, test_X = load_data("../datasets/agoda_cancellation_train.csv",
                                         "test_set_week_1.csv")

    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(train_X, train_y)

    # Store model predictions over test set
    evaluate_and_export(estimator, test_X, "318421476_318636420_208780957.csv")
