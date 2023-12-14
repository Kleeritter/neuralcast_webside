def neural_forecast(variable):
    import xarray as xr
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torch.nn as nn
    import pytorch_lightning as pl
    import numpy as np
    import random
    import xarray as xr
    from scipy import stats
    from datetime import timedelta
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.preprocessing import MinMaxScaler


    dataset = xr.open_dataset('/Users/alex/Code/neuralcast_webside/latest_herrenhausen_normal.nc')
    # Die Zeitdimension extrahieren
    time_dimension = dataset['time']
    #print(time_dimension)

    # Das aktuelle Zeitpunkt erhalten (letzter Zeitpunkt in der Datei)
    current_time = time_dimension[-1].values

    # Die Startzeitpunkt für die letzten 72 Stunden berechnen
   #start_time = current_time - timedelta(hours=24)
    start_time = current_time - np.timedelta64(23, 'h')

    # Die Daten in der Zeitdimension slicen
    sliced_dataset = dataset.sel(time=slice(start_time, current_time))
    dataset.close()
    print(sliced_dataset.herrenhausen_Temperatur.values)
    modelpath= "converting/webside_training/saved_models/multi/best_model_state_herrenhausen_Temperatur.pt"
    hyper_params_path= "converting/webside_training/webside_params/multi/best_params_lstm_multi_herrenhausen_Temperatur.yaml"
    forecasts= multilstm_full(modell=modelpath,data=sliced_dataset,forecast_horizon=24,forecast_var="herrehausen_Temperatur",hyper_params_path=hyper_params_path)
    print(forecasts)
    dataprint= np.append(sliced_dataset.herrenhausen_Temperatur.values,forecasts)
    print(dataprint)

    # Beispielwerte
    dataa = np.arange(-20.0, 45.0)  # Annahme: Ein eindimensionaler Datensatz


    scalera = MinMaxScaler()
    scalera.fit(dataa.reshape(-1, 1))
    denormalized_values = scalera.inverse_transform(dataprint.reshape(-1, 1)).flatten()

    plt.plot(denormalized_values)
    plt.show()
   
    return