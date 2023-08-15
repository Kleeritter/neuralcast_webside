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
import logging
from prophet import Prophet
import logging

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled = True
#forecast_var = "diffuscmp11"
window_size=672
forecast_horizont=24


#forecast_horizonts=[24]#[2,4,6,12,15,18,24,32,48,60,72,84,96,192]
#window_sizes=[672]#[16*24*7,8*7*24,4*7*24,2*7*24,7*24,6*24,5*24,4*24,3*24,2*24,24,12,6,3]
forecast_vars=["temp","press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50",     "rain", "wind_10", "wind_50","wind_dir_50_sin", "wind_dir_50_cos"]
#forecast_vars=["temp"]
random.seed(42)
def ppp(number):
    #print("start")
    forecast_var=forecast_vars[number]
    data = xr.open_dataset('../Data/zusammengefasste_datei_2016-2022.nc').to_dataframe()[
        [forecast_var]]  # [['index','temp']].to_dataframe()
    series = TimeSeries.from_dataframe(data, value_cols=forecast_var, freq="h")
    train, series = series.split_before(pd.Timestamp("2020-01-01 00:00:00"))
    #forecast_horizonts=[2,4,6,12,15,18,24,32,48,60,72,84,96,192]
    #window_sizes=[16*4*7,8*7*24,4*7*24,2*7*24,7*24,6*24,5*24,4*24,3*24,2*24,24,12,6,3]
    forecast_horizont=24#permutations[number][0] #forecast_horizonts[-number]
    window_size=168#empfohlene_fenstergroessen[forecast_var]#permutations[number][1]#window_sizes[number]


    startday = datetime.datetime(2022, 1, 1, 0) - datetime.timedelta(
        hours=window_size)  # .strftime('%Y-%m-%d %H:%M:%S')
    rest, year_values = series.split_after(pd.Timestamp(startday))

    year_values= year_values.pd_dataframe()
    year_values = year_values.reset_index().rename(columns={'index': 'ds', forecast_var: 'y'})
    #print(year_values)

    #train, val = rest.split_before(0.8)
    #plot_acf(train, m=12, alpha=0.05)
    #plt.show()
    #for m in range(2, 48):
     #   is_seasonal, period = check_seasonality(year_values, m=m, alpha=0.05)
      #  if is_seasonal:
       #     print("There is seasonality of order {}.".format(period))

    for window, last_window in tqdm(zip(range(window_size, len(year_values), forecast_horizont),
                                   range(0, len(year_values) - window_size, forecast_horizont))):
        #model =NaiveMovingAverage(input_chunk_length=168)# ARIMA(p=0,d=1,q=1,seasonal_order=(0,1,1,24))#StatsForecastAutoARIMA(season_length=24)
        #model.fit(year_values[last_window:window])
        #print(model)
        model = Prophet()
        model.fit(year_values[last_window:window])
        if last_window == 0:
           # pred_tmp = model.predict( n=forecast_horizont, verbose=False)
           future = model.make_future_dataframe(periods=forecast_horizont, freq='H')
           dfs= model.predict(future)
           forecast_values = dfs[['ds', 'yhat']].tail(24)
           forecast_values = forecast_values.set_index('ds')
           forecast_values=forecast_values.rename(columns={'yhat': forecast_var})

           df = forecast_values
           #print(df)
            #df = pred_tmp.pd_dataframe()
            #print(df)

        else:
            #pres = model.predict( n=forecast_horizont, verbose=False)
            future = model.make_future_dataframe(periods=forecast_horizont, freq='H')
            dfs = model.predict(future)
            forecast_values = dfs[['ds', 'yhat']].tail(24)
            forecast_values = forecast_values.set_index('ds')
            forecast_values = forecast_values.rename(columns={'yhat': forecast_var})
            fs = forecast_values
            #fs = pres.pd_dataframe()
            df = pd.concat([df, fs])

    if len(df) > 8760:
        df = df[:-(len(df)-8760)]
    #preds=TimeSeries.from_dataframe(df)
    #preds.plot(label="Forecast")
    #val.plot(label="Obs")
    #plt.show()
    file= "../Visualistion/ppp/"+forecast_var+str(window_size)+"_"+str(forecast_horizont)+".nc"
    #output=preds.pd_dataframe().to_xarray().to_netcdf(file)
    output=df.to_xarray().to_netcdf(file)

numbers=range(0,len(forecast_vars))
with Pool(6) as p:
    p.map(ppp, numbers)

#ppp(0)