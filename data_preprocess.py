import pandas as pd
import numpy as np
from datetime import datetime
from functools import reduce
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

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
Seattle['weather_description'] = Seattle['weather_description'].astype('category')
Seattle.dropna(inplace=True)
print(Seattle.index)


features_ori = list(Seattle)
features_ori.remove('weather_description')
# features_ori.remove('temperature')
scaler = MinMaxScaler()
Seattle[features_ori] = scaler.fit_transform(Seattle[features_ori])

def series_to_supervised(data, n_in=0, n_out=3, dropnan=True):
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


#######################################################################################################################
data_X = []
data_Y = []
sequence_length = 48
temp_features = Seattle[features_ori].copy()
temp_target = Seattle['temperature'].copy()
for index in range(len(Seattle.index) - sequence_length):
    data_X.append(temp_features.iloc[index:index+sequence_length].values)
    data_Y.append(temp_target.iloc[index:index+sequence_length].values)
data_X = np.array(data_X)
data_Y = np.array(data_Y)
train_index = 365*24
val_index = 31*24
test_index = 62*24
train_X = data_X[:train_index, :-1, :]
train_Y = data_Y[:train_index, -1]
val_X = data_X[train_index:train_index+val_index, :-1, :]
val_Y = data_Y[train_index:train_index+val_index, -1]
test_X = data_X[train_index+val_index:train_index+val_index+test_index, :-1, :]
test_Y = data_Y[train_index+val_index:train_index+val_index+test_index, -1]

model = Sequential()
model.add(LSTM(50, input_shape=(None, data_X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

history = model.fit(train_X, train_Y, epochs=20, batch_size=72, validation_data=(val_X, val_Y), verbose=2, shuffle=False)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

y_pred = model.predict(test_X)
y_pred = y_pred.reshape(y_pred.shape[0],)
rmse = np.sqrt(mean_squared_error(test_Y, y_pred))
plt.figure()
plt.plot(y_pred)
plt.plot(test_Y)
plt.show()
print('Test RMSE: %.3f' % rmse)


# #######################################################################################################################
# # plt.figure()
# # Seattle.plot(subplots=True)
# # plt.show()
# Seattle_final = series_to_supervised(Seattle)
# # print(Seattle_final)
# # Seattle_final = pd.get_dummies(Seattle_final, columns=['weather_description(t)', 'weather_description(t+1)'])
# features = list(Seattle_final)
# features.remove('temperature(t)')
# features.remove('weather_description(t)')
# features.remove('weather_description(t+1)')
# features.remove('weather_description(t+2)')
# target = 'temperature(t)'
# N_TRAIN_DAYS = 365
# N_TRAIN_HOURS = N_TRAIN_DAYS * 24
#
# train_X = Seattle_final.loc[:'2015-10-01 12:00:00', features].values
# train_Y = Seattle_final.loc[:'2015-10-01 12:00:00', target].values
# val_X = Seattle_final.loc['2015-10-01 12:00:00':'2015-11-01 12:00:00', features].values
# val_Y = Seattle_final.loc['2015-10-01 12:00:00':'2015-11-01 12:00:00', target].values
# test_X = Seattle_final.loc['2015-11-01 12:00:00':'2016-01-01 12:00:00', features].values
# test_Y = Seattle_final.loc['2015-11-01 12:00:00':'2016-01-01 12:00:00', target].values
#
# train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1])
# val_X = val_X.reshape(val_X.shape[0], 1, val_X.shape[1])
# test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])
#
# model = Sequential()
# # model.add(LSTM(40, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(LSTM(40, return_sequences=True))
# model.add(LSTM(40))
# model.add(Dense(1))
# model.compile(loss='mae', optimizer='adam')
# history = model.fit(train_X, train_Y, epochs=50, batch_size=72, validation_data=(val_X, val_Y), verbose=2, shuffle=False)
#
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()

#
# y_pred = model.predict(test_X)
# y_pred=y_pred.reshape(y_pred.shape[0],)
# rmse = np.sqrt(mean_squared_error(test_Y, y_pred))
# plt.figure()
# plt.plot(y_pred)
# plt.plot(test_Y)
# plt.show()
# print('Test RMSE: %.3f' % rmse)
# #######################################################################################################################