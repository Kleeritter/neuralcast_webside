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
references_path="../forecast_sarima.nc"
lstm_uni_path="../forecast_lstm_uni.nc"
lstm_multi_path="../time_test_better_a.nc"#"../../Model/timetest/lstm_multi/output/temp/timetest_lstm_multitemp_24_6.nc"#"forecast_lstm_multi.nc"
tft_path="../tft_dart.nc"
nc_path = '../../Data/stunden/'+str(forecast_year)+'_resample_stunden.nc' # Replace with the actual path to your NetCDF file
nhits="../nhit.nc"
lstm_uni=xr.open_dataset(lstm_uni_path).to_dataframe()
lstm_multi=xr.open_dataset(lstm_multi_path).to_dataframe()
tft=xr.open_dataset(tft_path).to_dataframe()
data = xr.open_dataset(nc_path).to_dataframe()
references= xr.open_dataset(references_path).to_dataframe()
nhits=xr.open_dataset(nhits).to_dataframe()
daystart=pd.to_datetime(str(forecast_year)+"-12-08 00:00")
dayend=pd.to_datetime(str(forecast_year)+"-12-20 23:00")


sns.set_theme(style="darkgrid")
temps = pd.DataFrame({
    'Date': data.loc[daystart:dayend].index,
    'T_gemmessen':data.loc[daystart:dayend]["temp"],
    'T_LSTM_Multi': lstm_multi.loc[daystart:dayend]["temp"],
    'T_LSTM_UNI':lstm_uni.loc[daystart:dayend]["temp"],
    'SARIMA': references.loc[daystart:dayend]["temp"],
    'T_Nhits':nhits.loc[daystart:dayend]["temp"]


})

sns.lineplot(x="Date", y='value', hue='variable',data=pd.melt(temps, ['Date']))#,ax=ax[1,0])
#sns.despine(offset=10, trim=True)

plt.show()