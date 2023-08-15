import numpy as np
import matplotlib
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import math
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.dates as mdates
forecast_year=2022
references_path="../auto_arima.nc"
lstm_uni_path="../time_test_single.nc"
lstm_multi_path="../time_test_better_a.nc"#"../forecast_lstm_multi.nc"#"../../Model/timetest/lstm_multi/output/temp/timetest_lstm_multitemp_24_24.nc"#"forecast_lstm_multi.nc"
tft_path="../tft_dart.nc"
nc_path = '../../Data/stunden/'+str(forecast_year)+'_resample_stunden.nc' # Replace with the actual path to your NetCDF file
nhits="../nhit.nc"
baseline="../../Model/baseline/baseline_n.nc"
lstm_multi_cor ="../cortest_all_p.nc"

lstm_uni=xr.open_dataset(lstm_uni_path).to_dataframe()
lstm_multi=xr.open_dataset(lstm_multi_path).to_dataframe()
tft=xr.open_dataset(tft_path).to_dataframe()
data = xr.open_dataset(nc_path).to_dataframe()
baseline=xr.open_dataset(baseline).to_dataframe()
references= xr.open_dataset(references_path).to_dataframe()
nhits=xr.open_dataset(nhits).to_dataframe()
lstm_multi_cor=xr.open_dataset(lstm_multi_cor).to_dataframe()
daystart=pd.to_datetime(str(forecast_year)+"-02-15 00:00")
dayend=pd.to_datetime(str(forecast_year)+"-02-20 23:00")

fig, ax = plt.subplots(2, 2,figsize=(10,10),sharex=True)
def frame(var,cor=0):
    winds = pd.DataFrame({
        'Date': data.loc[daystart:dayend].index,
        'observed':data.loc[daystart:dayend][var]-cor,
        'multivariate LSTM': lstm_multi.loc[daystart:dayend][var]-cor,
        'univariate LSTM':lstm_uni.loc[daystart:dayend][var],
        'SARIMA': references.loc[daystart:dayend][var],
        'baseline':nhits.loc[daystart:dayend][var],
        'cor. multi. LSTM':lstm_multi_cor.loc[daystart:dayend][var]
    })
    return winds
sns.set_context("talk")
for i,j in zip(range(0,2),range(0,2)):
    ax[i,j].grid(True)

ax[0,0].grid(True)
ax[0,1].grid(True)
ax[1,0].grid(True)
ax[1,1].grid(True)
#ax[1,0].grid(True)
#sns.set_theme(style="darkgrid")
sns.lineplot(x="Date", y='value', hue='variable',data=pd.melt(frame(var="wind_50"), ['Date']),ax=ax[0,0],legend=False)
sns.lineplot(x="Date", y='value', hue='variable',data=pd.melt(frame(var="gust_50"), ['Date']),ax=ax[0,1],legend=False)
sns.lineplot(x="Date", y='value', hue='variable',data=pd.melt(frame(var="wind_10"), ['Date']),ax=ax[1,0],legend=False)
sns.lineplot(x="Date", y='value', hue='variable',data=pd.melt(frame(var="rain",cor=0), ['Date']),ax=ax[1,1],legend=False)

ax[0,0].set_ylabel('wind speed [m/s]')
ax[0,1].set_ylabel('wind gust [m/s]')
ax[1,0].set_ylabel('wind speed [m/s]')
ax[1,1].set_ylabel('precepitation [mm]')
date_format = mdates.DateFormatter('%d.%m')
ax[1,1].xaxis.set_major_formatter(date_format)
ax[1,1].legend( ['Line Up', 'Line Down'],loc='lower right', bbox_to_anchor=(1.2, 0.0),fancybox=True, shadow=True, ncol=1)

# Anzahl der x-Ticks reduzieren
# ax[1, 0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))  # Du kannst die Anzahl nach Bedarf ändern

# ax[1, 0].tick_params(rotation=45)
# Ticks alle 2 Jahre anzeigen
years_locator = mdates.DayLocator(interval=1)
ax[1,0].xaxis.set_major_locator(years_locator)
#plt.grid(True)
#plt.legend(loc='upper left')
#right_inset_ax = fig.add_axes([.15, .7, .1, .1])
#right_inset_ax.bar(rmse_data["Model"],rmse_data["RMSE"])
#sns.despine(offset=10, trim=True)
#plt.text(calc_rmse(lstm_multi), fontsize=12, color='red')
#plt.savefig("test.png")
plt.tight_layout()
plt.savefig("/home/alex/Dokumente/Bach/figures/storms.png", dpi=300)
plt.show()