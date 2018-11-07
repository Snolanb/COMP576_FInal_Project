import pandas as pd
import numpy as np
from datetime import datetime


DATA_DIR = './historical_hourly_weather_data/'
temperature = pd.read_csv(DATA_DIR+'temperature.csv')
cities = list(temperature)
cities.pop(0)
temperature['year'] = temperature.apply(lambda row: row.datetime[0:4], axis=1)
temperature['month'] = temperature.apply(lambda row: row.datetime[5:7], axis=1)
temperature['day'] = temperature.apply(lambda row: row.datetime[8:10], axis=1)
temperature['hour'] = temperature.apply(lambda row: row.datetime[11:13], axis=1)
temperature = temperature.set_index(['year', 'month', 'day', 'hour'])
temperature.drop('datetime', axis=1, inplace=True)
temperature[cities] = temperature[cities].applymap(lambda k: k * 9.0/5.0 - 459.67)
print(temperature.dtypes)
temperature.head()