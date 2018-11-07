import pandas as pd
import numpy as np
from datetime import datetime
from functools import reduce

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
Seattle = Seattle.set_index(['year', 'month', 'day', 'hour'])
Seattle.drop('datetime', axis=1, inplace=True)

print(Seattle.index)