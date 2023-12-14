import numpy as np

def resample(netcdf_filepath, outputfile, v=2):
    import xarray as xr
    import pandas as pd
    import glob
    import numpy as np
    from scipy.interpolate import interp1d
    ds=wind_split(netcdf_filepath)
    #ds = xr.open_dataset(netcdf_filepath)
    time_index = pd.to_datetime(ds['time'].values, unit='s')
    if v==1:
        vars= ["herrenhausen_Temperatur","herrenhausen_Druck","herrenhausen_Feuchte","dach_Diffus_CMP-11","dach_Global_CMP-11","herrenhausen_Gust_Speed", "sonic_Gust_Speed","herrenhausen_Regen","herrenhausen_Wind_Speed",
       "sonic_Wind_Speed","sonic_Wind_Dir_sin","sonic_Wind_Dir_cos","derived_Regen_event","derived_Taupunkt3h","derived_Temp3h", "derived_Press3h","derived_Press_sl","derived_Taupunkt","derived_rainsum3h","derived_vertwind" ]

        ds["herrenhausen_Temperatur"]= ds["herrenhausen_Temperatur"] +273.15
        ds["derived_Taupunkt"]= dew_pointa( ds["herrenhausen_Temperatur"], ds["herrenhausen_Feuchte"])
        print(ds["derived_Taupunkt"])
        #ds = press_reduction_international(ds) #ds["derived_Press_sl"]= pressreduction_international(ds["herrenhausen_Druck"],51,ds["herrenhausen_Temperatur"])     
        #ds["derived_Press_sl"]= #ds.apply(press_reduction_international,axis=1)
        ds["derived_Press_sl"]= pressreduction_international(ds["herrenhausen_Druck"],51,ds["herrenhausen_Temperatur"]) 
            # Calculate resampled variables
        ds["derived_Taupunkt3h"] = resample_var(ds, "derived_Taupunkt")
        ds["derived_Press3h"] = resample_var(ds, "derived_Press_sl")
        ds["derived_rainsum3h"] = resample_var(ds, "herrenhausen_Regen", sum=True)
        ds["derived_Temp3h"] = resample_var(ds, "herrenhausen_Temperatur")

        # Calculate additional variables
        ds["derived_vertwind"] = ds["sonic_Wind_Speed"] - ds["herrenhausen_Wind_Speed"]
        ds["derived_Regen_event"] = ds["herrenhausen_Regen"].rolling('3H').apply(lambda x: 1 if x.sum() > 0 else 0).fillna(0)
        #ds["rain"] = ds["rain"] + 1
    else:
        vars =["dach_CO2_ppm","dach_Diffus_CMP-11","dach_Geneigt_CM-11","dach_Global_CMP-11","herrenhausen_Druck","herrenhausen_Feuchte","herrenhausen_Gust_Speed","herrenhausen_Pyranometer_CM3","herrenhausen_Regen","herrenhausen_Temperatur","herrenhausen_Wind_Speed",
        "sonic_Gust_Speed","sonic_Temperatur","sonic_Wind_Dir_sin","sonic_Wind_Dir_cos","sonic_Wind_Speed"]
    
    values = ds[vars].isel(time=time_index.minute % 60 == 0)

    hourly_range= pd.date_range(start=time_index.min(), end=time_index.max(), freq='1H')
    dfs = pd.DataFrame(index=hourly_range)

    for var_name, var in values.variables.items():
            if var_name != "time" and values[var_name].isnull().all() != True:
                if var_name == "herrenhausen_Regen":
                    hourly_var = ds[var_name].resample(time='1H', origin="epoch").sum()
                    dfs[var_name] = hourly_var
                elif var_name == "wind_dir_50":
                    hourly_var = ds[var_name].resample(time='1H', origin="epoch").mean()
                    hourly_var[hourly_var < 0] = 0
                    #ds[var_name] = hourly_var
                    dfs[var_name] = hourly_var
                else:
                    hourly_var = ds[var_name].resample(time='1H', origin="epoch").mean()
                    #ds[var_name] = hourly_var
                    dfs[var_name] = hourly_var

    print(dfs)
    df_cleaned = dfs.interpolate(method='linear')
    df_cleaned.index.names = ['time']
    #if "wind_dir_50" in vars:
     #   df_cleaned.loc[df_cleaned['wind_dir_50'] < 0, 'wind_dir_50'] = 0
    df_cleaned.to_xarray().to_netcdf(outputfile)
    ds.close()

    return
def load_hyperparameters(file_path):
    import yaml
    with open(file_path, 'r') as file:
        hyperparameters = yaml.safe_load(file)
    return hyperparameters

def wind_split(file):
    import os
    import xarray as xr
    import math
    import pandas as pd
    import yaml

    # Open the NetCDF file and convert it to a DataFrame
    data = xr.open_dataset(file).to_dataframe()

    # Calculate sine and cosine of wind direction and add as new columns
    data["sonic_Wind_Dir_sin"] = data["sonic_Wind_Dir"].apply(lambda x: math.sin(math.radians(x)))
    data["sonic_Wind_Dir_cos"] = data["sonic_Wind_Dir"].apply(lambda x: math.cos(math.radians(x)))

    # Convert the DataFrame back to xarray format and save it to the same NetCDF file
   
    return  data.to_xarray()


def normalize(netcdf_filepath, outputfile, v=2):
    import xarray as xr
    import pandas as pd
    import glob
    import numpy as np
    import yaml
    import os 
    from sklearn.preprocessing import MinMaxScaler

    ds = xr.open_dataset(netcdf_filepath)
    data= ds.to_dataframe()
    for column in data.columns:
            values = data[column].values.reshape(-1, 1)

            scaler = MinMaxScaler(feature_range=(0, 1))
            #print(os.getcwd())
            if v==1:
                param_path ='../webside_training/params_for_normal_imuknet.yaml'  # "../../Data/params_for_normal.yaml"
            else:
                param_path ='../webside_training/params_for_normal_imuknet2.yaml'  # "../../Data/params_for_normal.yaml"
            params = load_hyperparameters(param_path)

            mins = params[column]['min']#params["Min_" + column]
            maxs = params[column]['max']#["Max_" + column]
            train_values = [mins, maxs]
            X_train_minmax = scaler.fit_transform(np.array(train_values).reshape(-1, 1))
            scaled_values = scaler.transform(values)
            data[column] = scaled_values.flatten()
    data.index.names = ['time']
    data.to_xarray().to_netcdf(outputfile)
    ds.close()
    return 
def resample_var(data, var, sum=False):
    data=data.to_dataframe()
    if sum:
        sample = data[var].rolling('3H').sum().fillna(0)
    else:
        sample = data[var].rolling('3H').mean().diff().fillna(0)
    return sample


def dew_pointa(T, RH):
    import numpy as np
    """Berechnet den Taupunkt in Grad Celsius mit der Goff-Gratch-Gleichung."""
    a = 7.5
    b = 237.3
    alpha = ((a * T) / (b + T)) + np.log10(RH)
    T_dp = (b * alpha) / (a - alpha)
    return T_dp

def pressreduction_international(p,height,t):
    kappa=1.402
    M=0.02896
    g= 9.81
    r=8.314
    #pmsml = round(ds.apply(lambda row: row[p_column] * (1 - ((kappa - 1) / kappa) * ((M * g * (-1 * row[height_column])) / (r * row[t_column])) )**(kappa / (kappa - 1)), axis=1), 2)
    pmsl= p*(1-((kappa -1)/kappa) *((M*g*(-1*height))/(r*t)))**(kappa/(kappa -1))
    return pmsl

#def press_reduction_international(df):
  #  kappa = 1.402
   # M = 0.02896
   # g = 9.81
    #r = 8.314
    #height=51
    
    # Berechne pmsl für jede Zeile im DataFrame
    #df['derived_Press_sl'] = round(df.apply(lambda row: row[herrenhausen_Druck] * (1 - ((kappa - 1) / kappa) * ((M * g * (-1 * height)) / (r * row["herrenhausen_Temperatur"])) )**(kappa / (kappa - 1))), 2)
    #df['derived_Press_sl'] = df.apply(lambda row: round(row["herrenhausen_Druck"] * (1 - ((kappa - 1) / kappa) * ((M * g * (-1 * height)) / (r * row["herrenhausen_Temperatur"])) )**(kappa / (kappa - 1)), 2))
    #return df
#def press_reduction_international(row):
 #   kappa = 1.402
  #  M = 0.02896
   # g = 9.81
   # r = 8.314
 #   height=51
 #   result = round(
 #       row["herrenhausen_Druck"] * (1 - ((kappa - 1) / kappa) * ((M * g * (-1 * height)) / (r * row["herrenhausen_Temperatur"])) )**(kappa / (kappa - 1)), 2
 #   )
 #   return result
    #return round( row["herrenhausen_Druck"] * (1 - ((kappa - 1) / kappa) * ((M * g * (-1 * height)) / (r * row["herrenhausen_Temperatur"])) )**(kappa / (kappa - 1)), 2)
#years=np.arange(2016,2023)

#for year in years:
 #       print(year)
        #filename = "converting/viktor/"+str(year)+".nc"
        #outputname_resample = "converting/webside_data/resampled/"+str(year)+".nc"
        #outputname_normal = "converting/webside_data/normalized/"+str(year)+".nc"
        #resample(filename, outputname_resample)
        #normalize(outputname_resample,outputname_normal)

#filename = "latest_herrenhausen.nc"
#outputname_resample = "latest_herrenhausen_resample.nc"
#outputname_normal = "latest_herrenhausen_normal.nc"
#resample(filename, outputname_resample)
#normalize(outputname_resample,outputname_normal)