import darts
import xarray as xr
from tqdm import tqdm
import numpy as np
import optuna
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts.models import NaiveDrift
from darts.utils.statistics import plot_acf, check_seasonality
from darts import TimeSeries,concatenate
from darts.models import NaiveSeasonal
from darts.metrics import mape,smape,rmse
import datetime
import itertools
import random
import yaml
from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.models import ExponentialSmoothing, TBATS, AutoARIMA, Theta, NBEATSModel,NHiTSModel,TFTModel,ARIMA
from multiprocessing import Pool
import logging
forecast_var = "temp"
window_size=672
forecast_horizont=24
data=xr.open_dataset('../Data/zusammengefasste_datei_2016-2022.nc').to_dataframe()[[ "temp"]]#[['index','temp']].to_dataframe()
series = TimeSeries.from_dataframe(data, value_cols=forecast_var, freq="h")
train,series=series.split_before(pd.Timestamp("2020-01-01 00:00:00"))
model = ARIMA()
model.fit(train)
forecast_horizonts=[2,4,6,12,15,18,24,32,48,60,72,84,96,192]
window_sizes=[16*24*7,8*7*24,4*7*24,2*7*24,7*24,6*24,5*24,4*24,3*24,2*24,24,12,6,3]

random.seed(42)
permutations = list(itertools.product(forecast_horizonts, window_sizes))
random.shuffle(permutations)
print(permutations)
#pred=model.predict(24,year_values[:672])
#print(pred)
def darima(number):
    print("start")
    #forecast_horizonts=[2,4,6,12,15,18,24,32,48,60,72,84,96,192]
    #window_sizes=[16*4*7,8*7*24,4*7*24,2*7*24,7*24,6*24,5*24,4*24,3*24,2*24,24,12,6,3]
    forecast_horizont=permutations[number][0] #forecast_horizonts[-number]
    window_size=permutations[number][1]#window_sizes[number]


    startday = datetime.datetime(2022, 1, 1, 0) - datetime.timedelta(
        hours=window_size)  # .strftime('%Y-%m-%d %H:%M:%S')
    rest, year_values = series.split_after(pd.Timestamp(startday))

    #train, val = rest.split_before(0.8)


    for window, last_window in zip(range(window_size, len(year_values), forecast_horizont),
                                   range(0, len(year_values) - window_size, forecast_horizont)):

        if last_window == 0:
            pred_tmp = model.predict(series=year_values[last_window:window], n=forecast_horizont, verbose=False)
            df = pred_tmp.pd_dataframe()
            #print(df)

        else:
            pres = model.predict(series=year_values[last_window:window], n=forecast_horizont, verbose=False)

            fs = pres.pd_dataframe()
            df = pd.concat([df, fs])

    if len(df) > 8760:
        df = df[:-(len(df)-8760)]
    preds=TimeSeries.from_dataframe(df)
    #preds.plot(label="Forecast")
    #val.plot(label="Obs")
    #plt.show()
    file= "../Visualistion/sarima/dart/"+forecast_var+"/"+str(window_size)+"_"+str(forecast_horizont)+".nc"
    output=preds.pd_dataframe().to_xarray().to_netcdf(file)

    return


#numbers=[1,2,3]
numbers=range(0,len(permutations))
with Pool(6) as p:
    p.map(darima, numbers)