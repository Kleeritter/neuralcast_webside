# Import necessary functions from the 'converterfuncs' module
from converterfuncs import pressurereduction, dew_pointa


# Define a function to split wind direction into sine and cosine components
def wind_split(file):
    import os
    import xarray as xr
    import math
    import pandas as pd
    import yaml

    # Open the NetCDF file and convert it to a DataFrame
    data = xr.open_dataset(file).to_dataframe()

    # Calculate sine and cosine of wind direction and add as new columns
    data["wind_dir_50_sin"] = data["wind_dir_50"].apply(lambda x: math.sin(math.radians(x)))
    data["wind_dir_50_cos"] = data["wind_dir_50"].apply(lambda x: math.cos(math.radians(x)))

    # Convert the DataFrame back to xarray format and save it to the same NetCDF file
    data = data.to_xarray()
    data.to_netcdf(file)
    return


# Define a function to perform rolling window resampling on a variable
def resample(data, var, sum=False):
    if sum:
        sample = data[var].rolling('3H').sum().fillna(0)
    else:
        sample = data[var].rolling('3H').mean().diff().fillna(0)
    return sample


# Define a function to combine wind direction components into a single angle
def combine_wind_direction(row):
    import math
    sin_val = row["wind_dir_50_sin"]
    cos_val = row["wind_dir_50_cos"]
    combined_angle_rad = math.atan2(sin_val, cos_val)
    combined_angle_deg = math.degrees(combined_angle_rad)
    if combined_angle_deg < 0:
        combined_angle_deg += 360
    return combined_angle_deg


# Define a function to calculate combined wind direction and update the data
def rewind(file):
    import xarray as xr
    import pandas as pd

    # Open the NetCDF file and convert it to a DataFrame
    data = xr.open_dataset(file).to_dataframe()

    # Apply the 'combine_wind_direction' function to calculate combined wind direction
    data["wind_dir_50"] = data.apply(combine_wind_direction, axis=1)

    # Convert the DataFrame back to xarray format and save it to the same NetCDF file
    data = data.to_xarray()
    data.to_netcdf(file)
    return


# Define a function to add derived variables for the 'Herrenhausen' dataset
def add_derived_herrenhausen(file):
    import xarray as xr
    import pandas as pd

    # Open the NetCDF file and convert it to a DataFrame
    data = xr.open_dataset(file).to_dataframe()

    # Calculate resampled variables
    data["Taupunkt3h"] = resample(data, "dewpoint_calc")
    data["press3h"] = resample(data, "press_sl")
    data["rainsum3h"] = resample(data, "rain", sum=True)
    data["temp3h"] = resample(data, "temp")

    # Calculate additional variables
    data["gradwind"] = data["wind_50"] - data["wind_10"]
    data["taupunkt"] = data["dewpoint_calc"]
    data["rain_event"] = data["rain"].rolling('3H').apply(lambda x: 1 if x.sum() > 0 else 0).fillna(0)
    data["rain"] = data["rain"] + 1

    # Convert the DataFrame back to xarray format and save it to the same NetCDF file
    data = data.to_xarray()
    data.to_netcdf(file)
    return


# Define a function to add derived variables for the 'Ruthe' dataset
def add_derived_ruthe(file):
    import xarray as xr
    import pandas as pd

    # Open the NetCDF file and convert it to a DataFrame
    data = xr.open_dataset(file).to_dataframe()

    # Print column names for debugging
    print(data.columns)

    # Calculate resampled variables
    data["Taupunkt3h"] = resample(data, "taupunkt")
    data["rainsum3h"] = resample(data, "rain", sum=True)
    data["temp3h"] = resample(data, "temp")

    # Add new variable 'gust_10' based on existing data
    data["gust_10"] = data["wind_10_max"]

    # Calculate 'taupunkt' using custom 'dew_pointa' function
    data["taupunkt"] = data.apply(lambda row: dew_pointa(row['temp'] - 273.15, row['humid']), axis=1) + 273.15

    # Calculate 'rain_event' variable using rolling window
    data["rain_event"] = data["rain"].rolling('3H').apply(lambda x: 1 if x.sum() > 0 else 0).fillna(0)

    # Add an 'index' variable
    data["index"] = data.index

    # Convert the DataFrame back to xarray format and save it to the same NetCDF file
    data = data.to_xarray()
    data.to_netcdf(file)
    return


# Call the 'add_derived_ruthe' function with a specific file name
add_derived_ruthe('zusammengefasste_datei_ruthe.nc')
