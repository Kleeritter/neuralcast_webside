def neural_forecast_var_multi(variable, dataset):
    import xarray as xr
    import torch
    from torch.utils.data import Dataset, DataLoader
    #import torchvision
    import torch.nn as nn
    import pytorch_lightning as pl
    import numpy as np
    import random
    import xarray as xr
    from scipy import stats
    from datetime import timedelta
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from sklearn.preprocessing import MinMaxScaler

    #print(os.getcwd())

    ### Pfade ###
    modelpath= "forecast/imuknet/models/multi/best_model_state_"+variable+"_24_24.pt"
    hyper_params_path= "forecast/imuknet/params/multi/best_params_lstm_multi_"+variable+".yaml"
    params_path= "webside_training/params_for_normal_imuknet.yaml"
    ### Pfade ###
    ### Daten einlesen ###
    dataset = xr.open_dataset(dataset)
    # Die Zeitdimension extrahieren
    time_dimension = dataset['time']
    #print(time_dimension)

    # Das aktuelle Zeitpunkt erhalten (letzter Zeitpunkt in der Datei)
    current_time = time_dimension[-1].values

    # Die Startzeitpunkt für die letzten 72 Stunden berechnen
    start_time = current_time - np.timedelta64(23, 'h')

    # Die Daten in der Zeitdimension slicen
    sliced_dataset = dataset.sel(time=slice(start_time, current_time))
    dataset.close()
    #print(sliced_dataset.herrenhausen_Temperatur.values)

    var_pars = load_hyperparameters(params_path)
    
    if variable == "herrenhausen_Temperatur" or variable== "herrenhausen_Regen" or variable=="sonic_Wind_Dir_sin" or variable== "sonic_Wind_Dir_cos":
        corvars= ["herrenhausen_Temperatur","derived_Press_sl","herrenhausen_Feuchte","dach_Diffus_CMP-11","dach_Global_CMP-11","herrenhausen_Gust_Speed", "sonic_Gust_Speed","herrenhausen_Regen","herrenhausen_Wind_Speed",
       "sonic_Wind_Speed","sonic_Wind_Dir_sin","sonic_Wind_Dir_cos","derived_Taupunkt","derived_Taupunkt3h","derived_Press3h", "derived_rainsum3h","derived_Temp3h","derived_vertwind","derived_Regen_event" ]
    else:
         corvars= ["herrenhausen_Temperatur","derived_Press_sl","herrenhausen_Feuchte","dach_Diffus_CMP-11","dach_Global_CMP-11","herrenhausen_Gust_Speed", "sonic_Gust_Speed","herrenhausen_Regen","herrenhausen_Wind_Speed",
       "sonic_Wind_Speed","sonic_Wind_Dir_sin","sonic_Wind_Dir_cos","derived_Taupunkt","derived_Taupunkt3h","derived_Press3h", "derived_rainsum3h","derived_Temp3h","derived_vertwind" ]
    ### Daten einlesen ###

    ### Modell aufrufen und Vorhersage erstellen ###
    forecasts= multilstm_full(modell=modelpath,data=sliced_dataset,forecast_horizon=24,forecast_var=variable,hyper_params_path=hyper_params_path, corvars=corvars)
    #print(forecasts)
    #dataprint= np.append(sliced_dataset.herrenhausen_Temperatur.values,forecasts)
    #print(dataprint)
    ### Modell aufrufen und Vorhersage erstellen ###


    ### Vorhersage denormalisieren ###
    minimum_var = var_pars[variable]["min"]
    maximum_var = var_pars[variable]["max"]
    dataa = np.arange(minimum_var, maximum_var)  #


    scalera = MinMaxScaler()
    scalera.fit(dataa.reshape(-1, 1))
    denormalized_values = scalera.inverse_transform(forecasts.reshape(-1, 1)).flatten()
    ### Vorhersage denormalisieren ###

    ### Werte als NetCDF ausgeben ###
     
    ### Werte als NetCDF ausgeben ###
    #plt.plot(denormalized_values)
    #plt.show()
   
    return denormalized_values


def multilstm_full(modell,data,forecast_horizon,forecast_var="herrenhausen_Temperatur",hyper_params_path="",corvars=[]):
    from forecast.imuknet.funcs.funcs_lstm_multi import TemperatureModel_multi_full
    #from webside_training.resample_and_normallize import load_hyperparameters
    import numpy as np
    import torch
    from scipy import stats
    checkpoint_path = modell
    checkpoint = torch.load(checkpoint_path)
    predicted_values = []

    best_params = load_hyperparameters(hyper_params_path)

    model = TemperatureModel_multi_full(hidden_size=best_params['hidden_size'], learning_rate=best_params['learning_rate'], weight_decay=best_params['weight_decay'],
                                    num_layers=best_params['num_layers'], weight_initializer=best_params['weight_initializer'],forecast_horizont=forecast_horizon,window_size=best_params['window_size'],numvars=len(corvars))
    
    model.load_state_dict(checkpoint)  # ['state_dict'])
    model.eval()
    sliding_window = data.to_dataframe()[corvars]
    sliding_window = np.expand_dims(sliding_window, axis=0)

    input_data = torch.from_numpy(np.array(sliding_window)).float()
    with torch.no_grad():
        predicted_value = model(input_data)
    predicted_values = np.array(predicted_value).flatten()



    return predicted_values    

def load_hyperparameters(file_path):
    import yaml
    with open(file_path, 'r') as file:
        hyperparameters = yaml.safe_load(file)
    return hyperparameters


def neural_forecast_multi(dataset, outputfile, time_start):
    import xarray as xr
    import pandas as pd 
    from datetime import datetime
    ds = xr.Dataset()#[("time",pd.date_range(start='1/1/2018', periods=24,freq="1H"))])
    ds["time"] = pd.date_range(start=time_start, periods=24,freq="1H")
    variable_list=["herrenhausen_Temperatur", "derived_Press_sl","herrenhausen_Feuchte", "dach_Diffus_CMP-11","dach_Global_CMP-11","herrenhausen_Gust_Speed",
     "sonic_Gust_Speed","herrenhausen_Regen", "herrenhausen_Wind_Speed", "sonic_Wind_Speed","sonic_Wind_Dir_sin", "sonic_Wind_Dir_cos"]
    for variable in variable_list:
        #print(variable)
               # Generiere die Variable mit der Funktion
        variable_data = neural_forecast_var_multi(variable=variable, dataset=dataset)

        # Füge die Variable zum Xarray-Dataset hinzu und weise die Zeitdimension zu
        ds[variable] = (("time",), variable_data)

    ds.to_netcdf(outputfile)
    return

def neural_forecast_var_single(variable, dataset):
    import xarray as xr
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torch.nn as nn
    #import torchvision
    import pytorch_lightning as pl
    import numpy as np
    import random
    import xarray as xr
    from scipy import stats
    from datetime import timedelta
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from sklearn.preprocessing import MinMaxScaler

    #print(os.getcwd())

    ### Pfade ###
    modelpath= "forecast/imuknet/models/single/best_model_state_"+variable+"_24_24.pt"
    hyper_params_path= "forecast/imuknet/params/single/best_params_lstm_single_"+variable+".yaml"
    params_path= "webside_training/params_for_normal_imuknet.yaml"
    ### Pfade ###
    ### Daten einlesen ###
    dataset = xr.open_dataset(dataset)
    # Die Zeitdimension extrahieren
    time_dimension = dataset['time']
    #print(time_dimension)

    # Das aktuelle Zeitpunkt erhalten (letzter Zeitpunkt in der Datei)
    current_time = time_dimension[-1].values

    # Die Startzeitpunkt für die letzten 72 Stunden berechnen
    start_time = current_time - np.timedelta64(23, 'h')

    print(start_time)

    # Die Daten in der Zeitdimension slicen
    sliced_dataset = dataset.sel(time=slice(start_time, current_time))
    dataset.close()
    #print(sliced_dataset.herrenhausen_Temperatur.values)

    var_pars = load_hyperparameters(params_path)
    

    ### Daten einlesen ###

    ### Modell aufrufen und Vorhersage erstellen ###
    forecasts= lstm_single(modell=modelpath,data=sliced_dataset,forecast_horizon=24,forecast_var=variable,hyper_params_path=hyper_params_path, variable=variable)
    #print(forecasts)
    #dataprint= np.append(sliced_dataset.herrenhausen_Temperatur.values,forecasts)
    #print(dataprint)
    ### Modell aufrufen und Vorhersage erstellen ###


    ### Vorhersage denormalisieren ###
    minimum_var = var_pars[variable]["min"]
    maximum_var = var_pars[variable]["max"]
    dataa = np.arange(minimum_var, maximum_var)  #


    scalera = MinMaxScaler()
    scalera.fit(dataa.reshape(-1, 1))
    denormalized_values = scalera.inverse_transform(forecasts.reshape(-1, 1)).flatten()
    ### Vorhersage denormalisieren ###

    ### Werte als NetCDF ausgeben ###
     
    ### Werte als NetCDF ausgeben ###
    #plt.plot(denormalized_values)
    #plt.show()
   
    return denormalized_values
def neural_forecast_single(dataset, outputfile, time_start,today):
    import xarray as xr
    import pandas as pd 
    from datetime import datetime
    import os
    import numpy as np

    variable_list=["herrenhausen_Temperatur", "derived_Press_sl","herrenhausen_Feuchte", "dach_Diffus_CMP-11","dach_Global_CMP-11","herrenhausen_Gust_Speed",
     "sonic_Gust_Speed","herrenhausen_Regen", "herrenhausen_Wind_Speed", "sonic_Wind_Speed","sonic_Wind_Dir_sin", "sonic_Wind_Dir_cos", "sonic_Wind_Dir"]


    numered_variable_names = [f"{variable}_{today.strftime('%H')}" for variable in variable_list]

    try:
        dold = xr.open_dataset(outputfile)
        os.remove(outputfile)
    except FileNotFoundError:
        dold = None
        print(f"The file {outputfile} does not exist. Creating a new dataset.")

    ds = xr.Dataset()#[("time",pd.date_range(start='1/1/2018', periods=24,freq="1H"))])
    ds["time"] = pd.date_range(start=time_start, periods=24,freq="1H")

    ds.attrs["last_forecast_hour"] = today.strftime('%H')



    for variable, numvar in zip(variable_list, numered_variable_names):
        if variable!= "sonic_Wind_Dir":
            #print(variable)
                # Generiere die Variable mit der Funktion
            variable_data = neural_forecast_var_single(variable=variable, dataset=dataset)

            # Füge die Variable zum Xarray-Dataset hinzu und weise die Zeitdimension zu
            ds[numvar] = (("time",), variable_data)
        else:
            ds[numvar] = np.arctan2(ds[numered_variable_names[-3]], ds[numered_variable_names[-2]])

                # Umrechnung von Radiant in Grad
            ds[numvar] = np.degrees(ds[numvar])
            # Korrektur negativer Gradzahlen
            ds[numvar] = (ds[numvar] + 360) % 360
            ds.drop_vars([numered_variable_names[-3], numered_variable_names[-2]])

    # Berechnung der Windrichtung aus den sin- und cos-Werten
# Berechnung der Windrichtung aus den sin- und cos-Werten


    if dold is not None:
        # Concatenate the new dataset (ds) with the old dataset (dold) along the time dimension
        combined_ds  = ds.combine_first(dold)#xr.concat([dold, ds], dim="time")
    else:
        combined_ds = ds

    combined_ds.to_netcdf(outputfile)

    print(combined_ds)


    #ds.to_netcdf(outputfile)

    #try:
     #   ds = xr.open_dataset(outputfile)
        #with xr.open_dataset(outputfile,mode='a') as ds:
                 #   for variable, numvar in zip(variable_list, numered_variable_names):
                        #print(variable)
                            # Generiere die Variable mit der Funktion
                   #     variable_data = neural_forecast_var_single(variable=variable, dataset=dataset)

                        # Füge die Variable zum Xarray-Dataset hinzu und weise die Zeitdimension zu
                      #  ds[numvar] = (("time",), variable_data)
                        
                   # print(ds)
                    #ds.close()
                    #ds.to_netcdf(outputfile)
      #  os.remove(outputfile)
    
       # for variable, numvar in zip(variable_list, numered_variable_names):
            #print(variable)
                # Generiere die Variable mit der Funktion
        #    variable_data = neural_forecast_var_single(variable=variable, dataset=dataset)

            # Füge die Variable zum Xarray-Dataset hinzu und weise die Zeitdimension zu
         #   ds[numvar] = (("time",), variable_data)

        #ds.to_netcdf(outputfile)
    #except Exception as er :
     #   print("Exception", er)
      #  ds = xr.Dataset()#[("time",pd.date_range(start='1/1/2018', periods=24,freq="1H"))])
#
 #       ds["time"] = pd.date_range(start=time_start, periods=24,freq="1H")


  #      for variable, numvar in zip(variable_list, numered_variable_names):
            #print(variable)
                # Generiere die Variable mit der Funktion
   #         variable_data = neural_forecast_var_single(variable=variable, dataset=dataset)

            # Füge die Variable zum Xarray-Dataset hinzu und weise die Zeitdimension zu
    #        ds[numvar] = (("time",), variable_data)
#
 #       ds.to_netcdf(outputfile)
    return

def lstm_single(modell,data,forecast_horizon,forecast_var="herrenhausen_Temperatur",hyper_params_path="", variable=""):
    from forecast.imuknet.funcs.funcs_lstm_single import TemperatureModel
    #from webside_training.resample_and_normallize import load_hyperparameters
    import numpy as np
    import torch
    from scipy import stats
    checkpoint_path = modell
    checkpoint = torch.load(checkpoint_path)
    predicted_values = []

    best_params = load_hyperparameters(hyper_params_path)

    model = TemperatureModel(hidden_size=best_params['hidden_size'], learning_rate=best_params['learning_rate'], weight_decay=best_params['weight_decay'],
                                    num_layers=best_params['num_layers'], weight_intiliazier=best_params['weight_intiliazier'],forecast_horizont=forecast_horizon,window_size=24)
    
    #print("gans")
    model.load_state_dict(checkpoint)  # ['state_dict'])
    model.eval()
    sliding_window = data.to_dataframe()[variable]
    sliding_window = np.expand_dims(sliding_window, axis=0)
    sliding_window = np.expand_dims(sliding_window, axis=2)


    input_data = torch.from_numpy(np.array(sliding_window)).float()
    #print(input_data.shape)
    with torch.no_grad():
        predicted_value = model(input_data)
    predicted_values = np.array(predicted_value).flatten()



    return predicted_values 