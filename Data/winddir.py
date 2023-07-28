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

def combine_wind_direction(row):
    import math
    sin_val = row["wind_dir_50_sin"]
    cos_val = row["wind_dir_50_cos"]
    combined_angle_rad = math.atan2(sin_val, cos_val)
    combined_angle_deg = math.degrees(combined_angle_rad)
    if combined_angle_deg < 0:
        combined_angle_deg += 360

    return combined_angle_deg

def rewind(file):
    import xarray as xr
    import pandas as pd
    data = xr.open_dataset(file).to_dataframe()
    data["wind_dir_50"] = data.apply(combine_wind_direction, axis=1)
    data=data.to_xarray()
    data.to_netcdf(file)
    return
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
    data["rain"] =data["rain"]+1
    data=data.to_xarray()
    data.to_netcdf(file)
    return

def add_cool_stuff_ruthe(file):
    import xarray as xr
    import pandas as pd
    data = xr.open_dataset(file).to_dataframe()
    data["Taupunkt3h"] = resammple(data, "tau")
    data["rainsum3h"] = resammple(data, "rain", sum=True)
    data["temp3h"] = resammple(data, "temp")
    data["taupunkt"] = data["tau"]
    data["rain_event"] = data["rain"].rolling('3H').apply(lambda x: 1 if x.sum() > 0 else 0).fillna(0)
    data["rain"] =data["rain"]
    data["index"] = data.index
    data=data.to_xarray()
    data.to_netcdf(file)
    return


#wind_split('stunden/2021_resample_stunden.nc')
#wind_split('zusammengefasste_datei_2016-2019.nc')
#wind_split('zusammengefasste_datei_2016-2021.nc')
#add_cool_stuff('zusammengefasste_datei_2016-2022.nc')
#add_cool_stuff('zusammengefasste_datei_2016-2021.nc')
#add_cool_stuff('stunden/2022_resample_stunden.nc')
#add_cool_stuff('stunden/2021_resample_stunden.nc')

#rewind('../Visualistion/baseline.nc')
#rewind('../Visualistion/prophet.nc')
#rewind('../Visualistion/time_test_better_a.nc')
#rewind('../Visualistion/cortest_all.nc')
#rewind('../Visualistion/nhit.nc')
#rewind('../Visualistion/forecast_lstm_uni.nc')
add_cool_stuff_ruthe('ruthe_2019_resample_stunden.nc')
add_cool_stuff_ruthe('ruthe_2020_resample_stunden.nc')