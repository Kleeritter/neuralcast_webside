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
from darts.models import ExponentialSmoothing, TBATS, AutoARIMA, Theta, NBEATSModel,NHiTSModel,TFTModel,ARIMA,StatsForecastAutoARIMA, NaiveMovingAverage
from multiprocessing import Pool
from Model.funcs.visualer_funcs import load_hyperparameters
import logging

logging.basicConfig(level=logging.INFO)

window_size=672
forecast_horizont=24



forecast_vars = ["wind_10","gust_10","rain"]
random.seed(42)


def darima(number, Ruthe=True):
    #print("start")
    forecast_var=forecast_vars[number]
    if Ruthe:
        data=xr.open_dataset('../Data/zusammengefasste_datei_ruthe.nc').to_dataframe()[[forecast_var]]
    else:
     data = xr.open_dataset('../Data/zusammengefasste_datei_2016-2022.nc').to_dataframe()[
         [forecast_var]]  # [['index','temp']].to_dataframe()
    #data[forecast_var]=np.log(data[forecast_var])
    series = TimeSeries.from_dataframe(data, value_cols=forecast_var, freq="h")

    forecast_horizont=24#permutations[number][0] #forecast_horizonts[-number]
    window_size=672#empfohlene_fenstergroessen[forecast_var]#permutations[number][1]#window_sizes[number]
    print(data.max())

    startday = datetime.datetime(2020, 1, 1, 0) - datetime.timedelta(
        hours=window_size)  # .strftime('%Y-%m-%d %H:%M:%S')
    rest, year_values = series.split_after(pd.Timestamp(startday))


    hyper_params_path = '../Model/arima_params/bestparams/best_sarima_params_temp'+'.yaml'

    best_params = load_hyperparameters(hyper_params_path)
    non_seasonal_params = best_params["non_seasonal_params"]
    seasonal_params = best_params["seasonal_params"]

    p=non_seasonal_params["p"]
    d=non_seasonal_params["d"]
    q=non_seasonal_params["q"]
    P=seasonal_params["P"]
    D=seasonal_params["D"]
    Q=seasonal_params["Q"]
    if P and D and Q ==0 :
        Seasonal=0
    else:
        Seasonal = 24# seasonal_params["Seasonal"]

    for window, last_window in tqdm(zip(range(window_size, len(year_values), forecast_horizont),
                                   range(0, len(year_values) - window_size, forecast_horizont))):


        model = ARIMA(p=p,d=d,q=q,seasonal_order=(P,D,Q,Seasonal))
        #model=NaiveMovingAverage(input_chunk_length=24)
        model.fit(year_values[last_window:window])
        #print(model)
        if last_window == 0:
            pred_tmp = model.predict( n=forecast_horizont, verbose=False)
            df = pred_tmp.pd_dataframe()


        else:
            pres = model.predict( n=forecast_horizont, verbose=False)

            fs = pres.pd_dataframe()
            df = pd.concat([df, fs])

    if len(df) > 8760:
        df = df[:-(len(df)-8760)]
    preds=TimeSeries.from_dataframe(df)

    output = preds.pd_dataframe()
    print(output.head())
    print(output.max())
    file= "../Visualistion/RUTHE_ARIMA/"+forecast_var+str(window_size)+"_"+str(forecast_horizont)+".nc"
    output=output.to_xarray().to_netcdf(file)

    return


#numbers=[1,2,3]
numbers=range(0,len(forecast_vars))
with Pool(6) as p:
    p.map(darima, numbers)

#darima(0)