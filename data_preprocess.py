import pandas as pd
import numpy as np
from datetime import datetime
from functools import reduce
import matplotlib.pyplot as plt

DATA_DIR = './historical_hourly_weather_data/'
temperature = pd.read_csv(DATA_DIR+'temperature.csv')
humidity = pd.read_csv(DATA_DIR+'humidity.csv')
pressure = pd.read_csv(DATA_DIR+'pressure.csv')
weather_description = pd.read_csv(DATA_DIR+'weather_description.csv')
wind_direction = pd.read_csv(DATA_DIR+'wind_direction.csv')
wind_speed = pd.read_csv(DATA_DIR+'wind_speed.csv')
cities = list(temperature)
cities.pop(0)

temperature[cities] = temperature[cities].applymap(lambda k: k * 9.0/5.0 - 459.67)
Seattle_temperature = temperature[['datetime', 'Seattle']].copy()
Seattle_temperature.columns = ['datetime', 'temperature']
Seattle_pressure = pressure[['datetime', 'Seattle']].copy()
Seattle_pressure.columns = ['datetime', 'pressure']
Seattle_humidity = humidity[['datetime', 'Seattle']].copy()
Seattle_humidity.columns = ['datetime', 'humidity']
Seattle_weather_description = weather_description[['datetime', 'Seattle']].copy()
Seattle_weather_description.columns = ['datetime', 'weather_description']
Seattle_wind_direction = wind_direction[['datetime', 'Seattle']].copy()
Seattle_wind_direction.columns = ['datetime', 'wind_direction']
Seattle_wind_speed = wind_speed[['datetime', 'Seattle']].copy()
Seattle_wind_speed.columns = ['datetime', 'wind_speed']

dfs = [Seattle_temperature, Seattle_pressure, Seattle_humidity, Seattle_weather_description, Seattle_wind_direction, Seattle_wind_speed]
Seattle = reduce(lambda left, right: pd.merge(left, right, on='datetime'), dfs)

Seattle['year'] = Seattle.apply(lambda row: row.datetime[0:4], axis=1)
Seattle['month'] = Seattle.apply(lambda row: row.datetime[5:7], axis=1)
Seattle['day'] = Seattle.apply(lambda row: row.datetime[8:10], axis=1)
Seattle['hour'] = Seattle.apply(lambda row: row.datetime[11:13], axis=1)
# Seattle = Seattle.set_index(['year', 'month', 'day', 'hour'])
# Seattle.drop('datetime', axis=1, inplace=True)
Seattle = Seattle.set_index('datetime')
print(Seattle.index)


def series_to_supervised(data, n_in=0, n_out=2, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    colnames = list(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(colnames[j]+'(t-%d)' % i) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(colnames[j]+'(t)') for j in range(n_vars)]
        else:
            names += [(colnames[j]+'(t+%d)' % i) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# plt.figure()
# Seattle.plot(subplots=True)
# plt.show()
Seattle_final = series_to_supervised(Seattle)
# print(Seattle_final)

features = list(Seattle_final)
features.remove('temperature(t)')
target = 'temperature(t)'
N_TRAIN_DAYS = 365
N_TRAIN_HOURS = N_TRAIN_DAYS * 24

train = Seattle_final.loc[:'2013-10-01 12:00:00', features].values
test = Seattle_final.loc['2013-10-01 12:00:00':'2014-01-01 12:00:00', target].values

