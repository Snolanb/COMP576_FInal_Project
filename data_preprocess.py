import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = './historical_hourly_weather_data/'
temperature = pd.read_csv(DATA_DIR+'temperature.csv')
cities = list(temperature)
cities.pop(0)
temperature[cities] = temperature[cities].applymap(lambda k: k * 9.0/5.0 - 459.67)
print(temperature.dtypes)
temperature.head()