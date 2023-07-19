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

def resammple(data,var,sum=False):
    if sum:
        sample = data[var].rolling('3H').sum().fillna(0)
    else:
        sample = data[var].rolling('3H').mean().diff().fillna(0)
    return sample
def add_cool_stuff(file):
    import xarray as xr
    import pandas as pd
    data = xr.open_dataset(file).to_dataframe()
    data["Taupunkt3h"] = resammple(data, "dewpoint_calc")
    data["press3h"] = resammple(data, "press_sl")
    data["rainsum3h"] = resammple(data, "rain", sum=True)
    data["temp3h"] = resammple(data, "temp")
    data["gradwind"] = data["wind_50"] - data["wind_10"]
    data["taupunkt"] = data["dewpoint_calc"]
    data["rain_event"] = data["rain"].rolling('3H').apply(lambda x: 1 if x.sum() > 0 else 0).fillna(0)
    data=data.to_xarray()
    data.to_netcdf(file)
    return

wind_split('stunden/2021_resample_stunden.nc')
#wind_split('zusammengefasste_datei_2016-2019.nc')
wind_split('zusammengefasste_datei_2016-2021.nc')
#add_cool_stuff('zusammengefasste_datei_2016-2022.nc')
add_cool_stuff('zusammengefasste_datei_2016-2021.nc')

add_cool_stuff('stunden/2021_resample_stunden.nc')