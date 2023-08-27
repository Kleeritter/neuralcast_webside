# Import necessary libraries
import xarray as xr
import pandas as pd
import glob
import numpy as np
from scipy.interpolate import interp1d

# List of input files
filelist = sorted(glob.glob("/home/alex/PycharmProjects/nerualcast/Data/einer/*.nc"))

# List of years
years = np.arange(2007, 2023, 1)

# Define a function to resample variables at a 10-minute interval
def resample_zehner(filelist, years):
    for i in range(len(filelist)):
        print(years[i])
        ds = xr.open_dataset(filelist[i])
        time_index = pd.to_datetime(ds['time'].values, unit='s')

        vars = ["humid", "temp", "press", "press_sl", "dewpoint_calc", "ptd", "ptm", "wind_10", "wind_50",
                "wind_dir_50", "gust_10", "gust_50", "rain", "globalrcm11"]
        values = ds[vars].isel(time=time_index.minute % 10 == 0)

        for var_name, var in values.variables.items():
            if var_name != "time" and values[var_name].isnull().all() != True:
                if var_name == "rain":
                    hourly_var = ds[var_name].resample(time='10T', origin="epoch").sum()
                    values[var_name] = hourly_var
                elif var_name == "wind_dir_50":
                    hourly_var = ds[var_name].resample(time='10T', origin="epoch").mean()
                    values[var_name] = xr.where(hourly_var < 0, 0, hourly_var)
                else:
                    hourly_var = ds[var_name].resample(time='10T', origin="epoch").mean()
                    values[var_name] = hourly_var

        values.to_netcdf("zehner/" + str(years[i]) + '_resample_zehner.nc')
        ds.close()
    return

# Define a function to resample variables at an hourly interval
def resample_stunden(filelist, years, vars):
    for i in range(len(filelist)):
        print(years[i])
        ds = xr.open_dataset(filelist[i])
        time_index = pd.to_datetime(ds['time'].values, unit='s')

        values = ds[vars].isel(time=time_index.minute % 60 == 0)

        # Creating a DataFrame to store hourly data
        start_date = str(years[i]) + '-01-01 00:00:00'
        end_date = str(years[i]) + '-12-31 23:00:00'
        hourly_range = pd.date_range(start=start_date, end=end_date, freq='H')
        dfs = pd.DataFrame(index=hourly_range)

        for var_name, var in values.variables.items():
            if var_name != "time" and values[var_name].isnull().all() != True:
                if var_name == "rain":
                    hourly_var = ds[var_name].resample(time='1H', origin="epoch").sum()
                    values[var_name] = hourly_var
                elif var_name == "wind_dir_50":
                    hourly_var = ds[var_name].resample(time='1H', origin="epoch").mean()
                    hourly_var[hourly_var < 0] = 0
                    ds[var_name] = hourly_var
                    values[var_name] = hourly_var
                else:
                    hourly_var = ds[var_name].resample(time='1H', origin="epoch").mean()
                    ds[var_name] = hourly_var
                    values[var_name] = hourly_var

        # Interpolate missing data and save as xarray dataset
        dfs = dfs.join(values.to_dataframe())
        df_cleaned = dfs.interpolate(method='linear')
        if "wind_dir_50" in vars:
            df_cleaned.loc[df_cleaned['wind_dir_50'] < 0, 'wind_dir_50'] = 0
        df_cleaned.to_xarray().to_netcdf("ruthe_" + str(years[i]) + '_resample_stunden.nc')
        ds.close()
    return

# Call the resampling function for specific years and variables
resample_stunden(["ruthe_2019.nc"], [2019], vars=["humid", "temp", "rain", "globalrcmp11", "wind_10", "wind_dir_10", "tau", "wind_10_max"])
resample_stunden(["ruthe_2020.nc"], [2020], vars=["humid", "temp", "rain", "globalrcmp11", "wind_10", "wind_dir_10", "tau", "wind_10_max"])
