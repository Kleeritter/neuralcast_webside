# Import necessary libraries
import pandas as pd
import numpy as np
import xarray as xr
import glob
from converterfuncs import dewpointer, dew_pointa, dewpoint_new

# Define paths and filenames
towerfolder = "/home/alex/Dokumente/ruthemast/2020/"
folder = "/home/alex/Dokumente/ruthe/2020/"
output = "ruthe_2020.nc"

# List CSV files in the specified folders
towerfiles = sorted(glob.glob(towerfolder + "*.csv"))
files = sorted(glob.glob(folder + "*.csv"))

# Create an empty DataFrame to store concatenated data
datas = pd.DataFrame()

# Loop through the tower files
for i in range(len(towerfiles)):
    # Read tower data from CSV file
    data = pd.read_csv(towerfiles[i], delimiter=";", encoding="latin-1", dayfirst=True)

    # Rename columns for consistency
    new_col_names = {
        "         Datum/Zeit": "time",
        "CO2 (15m)": "CO2_15",
        "CO2 (10m)": "CO2_10",
        "CO2 (2m )": "CO2_2",
        "WindDir (°)": "wind_dir_10",
        "WindSpeed(m/s)": "wind_10",
        "WindMax(m/s)": "wind_10_max"
    }
    data.rename(columns=new_col_names, inplace=True)

    # Convert time column to datetime and set it as index
    data["time"] = pd.to_datetime(data["time"], dayfirst=True)
    data = data.set_index("time")

    # Read supplementary data from CSV file
    data_sup = pd.read_csv(files[i], delimiter=";", encoding="latin-1")
    data_sup = data_sup.rename(columns=lambda x: x.strip())

    # Rename columns for consistency
    new_col_names = {
        "Datum/Zeit": "time",
        "Temp": "temp",
        "Feuchte": "humid",
        "Regen": "rain",
        "Globalstrahlung": "globalrcmp11"
    }
    data_sup.rename(columns=new_col_names, inplace=True)

    # Convert time column to datetime and set it as index
    data_sup["time"] = pd.to_datetime(data_sup["time"], dayfirst=True)
    data_sup = data_sup.set_index("time")

    # Merge supplementary data with tower data
    data["temp"] = data_sup["temp"] + 273.15
    data["humid"] = data_sup["humid"].apply(lambda x: x if x < 1 else x / 1000)
    data["rain"] = data_sup["rain"]
    try:
        data["globalrcmp11"] = data_sup["globalrcmp11"]
    except:
        pass
    data["tau"] = data.apply(lambda row: dewpoint_new(row['temp'], row['humid']),
                             axis=1)  # dewpointer(data["temp"],data["humid"])

    # Concatenate the data to the 'datas' DataFrame
    datas = pd.concat([datas, data])

# Print the tail of the concatenated DataFrame for debugging
print(datas.tail())

# Convert the concatenated DataFrame to xarray format and save as a NetCDF file
datas.to_xarray().to_netcdf(output)
