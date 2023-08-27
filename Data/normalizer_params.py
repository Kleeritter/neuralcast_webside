from netCDF4 import Dataset
import os
import xarray as xr
import math
import pandas as pd
import yaml

data= xr.open_dataset('zusammengefasste_datei_2016-2022.nc').to_dataframe()

print(data.head())



max_values = {
    "Max_temp": round(max(data['temp']), 2),
    "Max_humid": round(max(data['humid']), 2),
    "Max_press_sl": round(max(data['press_sl']), 2),
    "Max_wind_50": round(max(data['wind_50']), 2),
    "Max_wind_10": round(max(data['wind_10']), 2),
    "Max_gust_50": round(max(data['gust_50']), 2),
    "Max_gust_10": round(max(data['gust_10']), 2),
    "Max_rain": round(max(data['rain']), 2),
    "Max_diffuscmp11": round(max(data['diffuscmp11']), 2),
    "Max_globalrcmp11": round(max(data['globalrcmp11']), 2),
    "Max_wind_dir_50_sin": round(max(data['wind_dir_50_sin']), 2),
    "Max_wind_dir_50_cos": round(max(data['wind_dir_50_cos']), 2),
    "Max_taupunkt": round(max(data['taupunkt']), 2),
    "Max_gradwind": round(max(data['gradwind']), 2),
    "Max_temp3h": round(max(data['temp3h']), 2),
    "Max_rainsum3h": round(max(data['rainsum3h']), 2),
    "Max_press3h": round(max(data['press3h']), 2),
    "Max_Taupunkt3h": round(max(data['Taupunkt3h']), 2)
}

# Calculate minimum values
min_values = {
    "Min_temp": round(min(data['temp']), 2),
    "Min_humid": round(min(data['humid']), 2),
    "Min_press_sl": round(min(data['press_sl']), 2),
    "Min_wind_50": round(min(data['wind_50']), 2),
    "Min_wind_10": round(min(data['wind_10']), 2),
    "Min_gust_50": round(min(data['gust_50']), 2),
    "Min_gust_10": round(min(data['gust_10']), 2),
    "Min_rain": round(min(data['rain']), 2),
    "Min_diffuscmp11": round(min(data['diffuscmp11']), 2),
    "Min_globalrcmp11": round(min(data['globalrcmp11']), 2),
    "Min_wind_dir_50_sin": round(min(data['wind_dir_50_sin']), 2),
    "Min_wind_dir_50_cos": round(min(data['wind_dir_50_cos']), 2),
    "Min_taupunkt": round(min(data['taupunkt']), 2),
    "Min_gradwind": round(min(data['gradwind']), 2),
    "Min_temp3h": round(min(data['temp3h']), 2),
    "Min_rainsum3h": round(min(data['rainsum3h']), 2),
    "Min_press3h": round(min(data['press3h']), 2),
    "Min_Taupunkt3h": round(min(data['Taupunkt3h']), 2)
}

# Combine both dictionaries into a single YAML dictionary
yaml_dict = {**max_values, **min_values}


with open('params_for_normal.yaml', 'w') as file:
    yaml.dump(yaml_dict, file)

