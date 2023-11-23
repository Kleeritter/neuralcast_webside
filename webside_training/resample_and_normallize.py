def resample(netcdf_filepath, outputfile):
    import xarray as xr
    import pandas as pd
    import glob
    import numpy as np
    from scipy.interpolate import interp1d

    ds = xr.open_dataset(netcdf_filepath)
    time_index = pd.to_datetime(ds['index'].values, unit='s')
    vars =["dach_CO2_ppm","dach_CO2_Sensor","dach_Diffus_CMP-11","dach_Geneigt_CM-11","dach_Global_CMP-11",
    "dach_Temp_AMUDIS_Box","dach_Temp_Bentham_Box","herrenhausen_Druck","herrenhausen_Feuchte","herrenhausen_Gust_Speed"
    ,"herrenhausen_Psychro_T","herrenhausen_Psychro_Tf","herrenhausen_Pyranometer_CM3","herrenhausen_Regen","herrenhausen_Temperatur","herrenhausen_Wind_Speed",
    "sonic_Gust_Speed","sonic_Temperatur","sonic_Wind_Dir","sonic_Wind_Speed"]
    
    values = ds[vars].isel(index=time_index.minute % 60 == 0)

    hourly_range= pd.date_range(start=time_index.min(), end=time_index.max(), freq='1H')
    dfs = pd.DataFrame(index=hourly_range)

    for var_name, var in values.variables.items():
            if var_name != "index" and values[var_name].isnull().all() != True:
                if var_name == "rain":
                    hourly_var = ds[var_name].resample(index='1H', origin="epoch").sum()
                    dfs[var_name] = hourly_var
                elif var_name == "wind_dir_50":
                    hourly_var = ds[var_name].resample(index='1H', origin="epoch").mean()
                    hourly_var[hourly_var < 0] = 0
                    #ds[var_name] = hourly_var
                    dfs[var_name] = hourly_var
                else:
                    hourly_var = ds[var_name].resample(index='1H', origin="epoch").mean()
                    #ds[var_name] = hourly_var
                    dfs[var_name] = hourly_var

    print(dfs)
    df_cleaned = dfs.interpolate(method='linear')
    if "wind_dir_50" in vars:
        df_cleaned.loc[df_cleaned['wind_dir_50'] < 0, 'wind_dir_50'] = 0
    df_cleaned.to_xarray().to_netcdf(outputfile)
    ds.close()

    return


def normalize():

    return 

resample("converting/viktor/2016.nc", "test.nc")