import torch
from sklearn.model_selection import train_test_split
from funcs.funcs_lstm_single  import TemperatureDataset
from funcs.funcs_lstm_multi import TemperatureDataset_multi
import pytorch_lightning as pl
import numpy as np
import random
import xarray as xr
pl.seed_everything(42)


# Setze den Random Seed für torch
torch.manual_seed(42)

# Setze den Random Seed für random
random.seed(42)

# Setze den Random Seed für numpy
np.random.seed(42)
storage="/home/alex/Dokumente/storage"
lite = '../Data/stunden/2016_resample_stunden.nc'
full = '../Data/zusammengefasste_datei_2016-2019.nc'
forecast_vars = ["wind_dir_50_sin","wind_dir_50_cos"]#, 'temp', "press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50",     "rain", "wind_10", "wind_50"]
#forecast_vars = ["wind_dir_50"]
lisa=["sin","cos"]
def create_single_trains():

    window_sizes= [24*7*4]
    file_path =full
    for forecast_var in forecast_vars:
        print(forecast_var)
        for window_size in window_sizes:
            training_data_path = storage+'/training/lstm_uni/train_'+forecast_var+"_"+str(window_size)+'.pt'
            val_data_path = storage+'/validation/lstm_uni/val_'+forecast_var+"_"+str(window_size)+'.pt'
            dataset = TemperatureDataset(file_path,window_size=window_size,forecast_horizont=24,forecast_var=forecast_var)
            train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)
            torch.save(train_data, training_data_path)
            torch.save(val_data, val_data_path)
            print("saved")
    return




def create_multi_trains():
    #forecast_vars=["wind_dir_50",'temp',"press_sl","humid","diffuscmp11","globalrcmp11","gust_10","gust_50", "rain", "wind_10", "wind_50"]
    window_sizes= [24*7*4]
    file_path = full
    for forecast_var in forecast_vars:
        print(forecast_var)
        for window_size in window_sizes:
            training_data_path = storage+'/training/lstm_multi/train_' + forecast_var + "_" + str(window_size) + '.pt'
            val_data_path = storage+'/validation/lstm_multi/val_' + forecast_var + "_" + str(window_size) + '.pt'
            dataset = TemperatureDataset_multi(file_path,window_size=window_size,forecast_horizont=24,forecast_var=forecast_var)
            train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)
            torch.save(train_data, training_data_path)
            torch.save(val_data, val_data_path)
            print("saved")
    return


#create_multi_trains()
create_single_trains()