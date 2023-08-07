import numpy as np
import matplotlib
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import math
from sklearn.metrics import mean_squared_error
import seaborn as sns

forecast_year=2022
references_path="../auto_arima.nc"
lstm_uni_path="../forecast_lstm_uni.nc"
lstm_multi_path="../time_test_better_a.nc"#"../forecast_lstm_multi.nc"#"../../Model/timetest/lstm_multi/output/temp/timetest_lstm_multitemp_24_24.nc"#"forecast_lstm_multi.nc"
tft_path="../tft_dart.nc"
nc_path = '../../Data/stunden/'+str(forecast_year)+'_resample_stunden.nc' # Replace with the actual path to your NetCDF file
nhits="../nhit.nc"

lstm_uni=xr.open_dataset(lstm_uni_path).to_dataframe()
lstm_multi=xr.open_dataset(lstm_multi_path).to_dataframe()
tft=xr.open_dataset(tft_path).to_dataframe()
data = xr.open_dataset(nc_path).to_dataframe()
references= xr.open_dataset(references_path).to_dataframe()
nhits=xr.open_dataset(nhits).to_dataframe()
daystart=pd.to_datetime(str(forecast_year)+"-02-15 00:00")
dayend=pd.to_datetime(str(forecast_year)+"-02-20 23:00")

fig, ax = plt.subplots(2, 2,figsize=(10,10))
def frame(var,cor=0):
    winds = pd.DataFrame({
        'Date': data.loc[daystart:dayend].index,
        'T_gemmessen':data.loc[daystart:dayend][var]-cor,
        'T_LSTM_Multi': lstm_multi.loc[daystart:dayend][var]-cor,
        'T_LSTM_UNI':lstm_uni.loc[daystart:dayend][var],
        'SARIMA': references.loc[daystart:dayend][var],
        'T_Nhits':nhits.loc[daystart:dayend][var]
    })
    return winds
sns.set_context("talk")

sns.set_theme(style="darkgrid")
sns.lineplot(x="Date", y='value', hue='variable',data=pd.melt(frame(var="wind_50"), ['Date']),ax=ax[0,0])
sns.lineplot(x="Date", y='value', hue='variable',data=pd.melt(frame(var="gust_50"), ['Date']),ax=ax[0,1])
sns.lineplot(x="Date", y='value', hue='variable',data=pd.melt(frame(var="wind_10"), ['Date']),ax=ax[1,0])
sns.lineplot(x="Date", y='value', hue='variable',data=pd.melt(frame(var="rain",cor=0), ['Date']),ax=ax[1,1])
plt.show()