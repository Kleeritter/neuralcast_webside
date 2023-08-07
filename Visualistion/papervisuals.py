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
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

import matplotlib.pyplot as plt
from darts.models import NaiveDrift
from darts.utils.statistics import plot_acf, check_seasonality
from darts import TimeSeries,concatenate
from darts.models import NaiveSeasonal
from darts.metrics import mape,smape,rmse
import datetime
import random
import yaml
from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.models import ExponentialSmoothing, TBATS, AutoARIMA, Theta, NBEATSModel,NHiTSModel,TFTModel,ARIMA,StatsForecastAutoARIMA, NaiveMovingAverage
from multiprocessing import Pool
from Model.funcs.visualer_funcs import load_hyperparameters
import logging
import seaborn as sns
def stationarity_duett():
    data = xr.open_dataset("../Data/stunden/2022_resample_stunden.nc").to_dataframe()
    forecast_var = "temp"
    series = TimeSeries.from_dataframe(data[24*130:24*130+(24*7)], value_cols=forecast_var, freq="h")
    plottingseries=TimeSeries.from_dataframe(data[24*130+24:24*130+(24*7)], value_cols=forecast_var, freq="h")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(8, 6),sharex=True)
    fig.suptitle('Seasonal differencing of the time series')
    ax[0].set_ylabel('temperature [K]')
    sns.lineplot(plottingseries.pd_dataframe(),color="black",ax=ax[0],legend=False)
    sns.scatterplot(plottingseries.pd_dataframe(),color="black",ax=ax[0],legend=False)

    periods=1
    diffs=series.diff()
    seasonal_diffs=diffs.diff(periods=24)
    sns.lineplot(seasonal_diffs.pd_dataframe(), color="black", ax=ax[1],legend=False)
    sns.scatterplot(seasonal_diffs.pd_dataframe(), color="black", ax=ax[1],legend=False)

    # Datumsformat anpassen
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H'))

    # Optional: Rotieren der Datumsbeschriftungen, um Überlappungen zu vermeiden
    plt.xticks(rotation=45)
    ax[1].set_ylabel('Δ temperature [K]')
    ax[1].set_xlabel('Time')
    plt.grid=True
    plt.legend=False
    #plt.show()

    # Plot speichern
    plt.savefig("/home/alex/Dokumente/Bach/figures/stationarity.png", dpi=300, bbox_inches="tight")
    return

def stationarity_mono(diff=False):
    data = xr.open_dataset("../Data/stunden/2022_resample_stunden.nc").to_dataframe()
    #print(data.head())
    forecast_var = "temp"
    series = TimeSeries.from_dataframe(data[24*130:24*130+72], value_cols=forecast_var, freq="h")
    plottingseries=TimeSeries.from_dataframe(data[24*130+24:24*130+72], value_cols=forecast_var, freq="h")
   # fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(8, 6),sharex=True)
    #plottingseries.plot(ax=ax[0])
   # fig.suptitle('Seasonal differencing of the time series')
    if diff==True:
        sns.lineplot(series.diff(periods=24).pd_dataframe(), color="black")
        plt.ylabel('seasonal temperature difference [K]')
    else:
        sns.lineplot(plottingseries.pd_dataframe(), color="black")
        plt.ylabel('Temperature [K]')
        plt.title("Time series with seasonal pattern")

    plt.xlabel('Date')
    plt.grid=True
    plt.show()
    return

def plot_relu():
    x=np.linspace(-1,1,100)
    y=np.maximum(x,0)
    fig,ax=plt.subplots(figsize=(8, 6))
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    sns.lineplot(x=x,y=y,color="black",ax=ax)
    #plt.plot(x,y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("ReLU")
    plt.grid(True)
    plt.savefig("/home/alex/Dokumente/Bach/figures/RELU.png", dpi=300)
    return



def mean_and_variance():
    data = xr.open_dataset("../Data/zusammengefasste_datei_2016-2021.nc").to_dataframe()
    des= data.describe()
    print(des)
    des.to_csv("/home/alex/Dokumente/Bach/figures/describe.csv")

    return

def temperature_plot():
    data = xr.open_dataset("../Data/zusammengefasste_datei_2016-2021.nc").to_dataframe()
    sns.set_context("paper", font_scale=1.5)
    sns.lineplot(data=data, x="index", y="temp", color="black")
    plt.xlabel("Date")
    plt.ylabel("Temperature [K]")
    plt.title("Temperature time series")
    plt.grid(True)
    plt.savefig("/home/alex/Dokumente/Bach/figures/temperature.png", dpi=300)
    return
#stationarity_duett()
def multiview():
    data = xr.open_dataset("../Data/zusammengefasste_datei_2016-2021.nc").to_dataframe()
    sns.set_context("paper", font_scale=1.5)

    fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(8, 6),sharex=True)

    fig.suptitle('parameters of the time series')
    sns.lineplot(data=data, x="index", y="temp", color="black",ax=ax[0,0],legend=False)
    sns.lineplot(data=data, x="index", y="humid", color="black",ax=ax[1,0],legend=False)
    sns.lineplot(data=data, x="index", y="wind_50", color="black",ax=ax[0,1],legend=False)
    sns.lineplot(data=data, x="index", y="rain", color="black",ax=ax[1,1],legend=False)
    ax[0,0].set_ylabel('temperature [K]')
    ax[1,0].set_ylabel('humidity [%]')
    ax[0,1].set_ylabel('wind speed [m/s]')
    ax[1,1].set_ylabel('precipitation [mm/h]')
    ax[1,0].set_xlabel('Date')
    # Anzahl der x-Ticks reduzieren
    #desired_num_ticks = 6  # Beispielwert
    date_format = mdates.DateFormatter('%Y')
    ax[1, 0].xaxis.set_major_formatter(date_format)

    # Anzahl der x-Ticks reduzieren
   # ax[1, 0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))  # Du kannst die Anzahl nach Bedarf ändern

    #ax[1, 0].tick_params(rotation=45)
    # Ticks alle 2 Jahre anzeigen
    years_locator = mdates.YearLocator(base=2)
    ax[1, 0].xaxis.set_major_locator(years_locator)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Platz für Titel
    plt.savefig("/home/alex/Dokumente/Bach/figures/multiview.png", dpi=300)
#stationarity_mono()

#plot_relu()

#mean_and_variance()
multiview()
#temperature_plot()