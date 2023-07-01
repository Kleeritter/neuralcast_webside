def wind_split(file):
    import os
    import xarray as xr
    import math
    import pandas as pd
    import yaml
    # Erstellen Sie eine neue NetCDF-Datei zum Zusammenfassen der Daten
    data = xr.open_dataset(file).to_dataframe()
    data["wind_dir_50_sin"]=data["wind_dir_50"].apply(lambda x: math.sin(math.radians(x)))
    data["wind_dir_50_cos"]=data["wind_dir_50"].apply(lambda x: math.cos(math.radians(x)))
    data=data.to_xarray()
    data.to_netcdf(file)
    return


wind_split('stunden/2022_resample_stunden.nc')
wind_split('zusammengefasste_datei_2016-2019.nc')
