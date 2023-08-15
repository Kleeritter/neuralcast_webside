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
references_path="../auto_arima.nc"#"../ppp/temp168_24.nc"#"../AUTOARIMA/temp672_24.nc"#"../forecast_sarima.nc"
lstm_uni_path="../time_test_single.nc"
lstm_multi_path="../time_test_better_a.nc"#"../../Model/timetest/lstm_multi/output/temp/timetest_lstm_multitemp_24_6.nc"#"forecast_lstm_multi.nc"
tft_path="../tft_dart.nc"
nc_path = '../../Data/stunden/'+str(forecast_year)+'_resample_stunden.nc' # Replace with the actual path to your NetCDF file
nhits="../nhit.nc"
baseline="../../Model/baseline/baseline_n.nc"
lstm_multi_cor ="../cortest_all_p.nc"
lstm_uni=xr.open_dataset(lstm_uni_path).to_dataframe()
lstm_multi=xr.open_dataset(lstm_multi_path).to_dataframe()
tft=xr.open_dataset(tft_path).to_dataframe()
data = xr.open_dataset(nc_path).to_dataframe()
references= xr.open_dataset(references_path).to_dataframe()
nhits=xr.open_dataset(nhits).to_dataframe()
baseline=xr.open_dataset(baseline).to_dataframe()
lstm_multi_cor=xr.open_dataset(lstm_multi_cor).to_dataframe()
daystart=pd.to_datetime(str(forecast_year)+"-07-18 00:00")
dayend=pd.to_datetime(str(forecast_year)+"-07-21 23:00")

def calc_rmse(model):
    return math.sqrt(mean_squared_error(data.loc[daystart:dayend]["temp"],model.loc[daystart:dayend]["temp"]))




#sns.set_theme(style="darkgrid")
#sns.set_context("talk")
temps = pd.DataFrame({
    'Date': data.loc[daystart:dayend].index,
    'observed':data.loc[daystart:dayend]["temp"],
    'multivariate LSTM': lstm_multi.loc[daystart:dayend]["temp"],
    'univariate LSTM':lstm_uni.loc[daystart:dayend]["temp"],
    'SARIMA': references.loc[daystart:dayend]["temp"],
    'seasonal naive':baseline.loc[daystart:dayend]["temp"],
    'cor. multi. LSTM':lstm_multi_cor.loc[daystart:dayend]["temp"]


})
rmse_data = pd.DataFrame({
    'Model': ['LSTM Multi', 'LSTM Uni', 'SARIMA', 'T_Nhits'],
    'RMSE': [calc_rmse(lstm_multi), calc_rmse(lstm_uni), calc_rmse(references), calc_rmse(nhits)]
})

# Erstelle einen FacetGrid für die Darstellung der RMSE-Werte
#g = sns.FacetGrid(rmse_data, aspect=1.5, height=5)
#sns.barplot(rmse_data["RMSE"], palette='pastel', order=['LSTM Multi', 'LSTM Uni', 'SARIMA', 'T_Nhits'])
#sns.set_context("paper", font_scale=1.5)
fig, ax = plt.subplots()#1, 2, figsize=(15, 10))
#sns.barplot(x="Model", y="RMSE", data=rmse_data,width=0.6)

sns.lineplot(x="Date", y='value', hue='variable',data=pd.melt(temps, ['Date']),ax=ax)
plt.xlabel("Date")
plt.ylabel("Temperature [K]")
plt.title("Temperature forecast for the heatwave in July 2022")
#plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d'))
date_format = mdates.DateFormatter('%d.%m')
ax.xaxis.set_major_formatter(date_format)
plt.legend(loc='lower right', bbox_to_anchor=(1.2, 0.0),fancybox=True, shadow=True, ncol=1)

# Anzahl der x-Ticks reduzieren
# ax[1, 0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))  # Du kannst die Anzahl nach Bedarf ändern

# ax[1, 0].tick_params(rotation=45)
# Ticks alle 2 Jahre anzeigen
years_locator = mdates.DayLocator(interval=1)
ax.xaxis.set_major_locator(years_locator)
plt.grid(True)
#plt.legend(loc='upper left')
#right_inset_ax = fig.add_axes([.15, .7, .1, .1])
#right_inset_ax.bar(rmse_data["Model"],rmse_data["RMSE"])
#sns.despine(offset=10, trim=True)
#plt.text(calc_rmse(lstm_multi), fontsize=12, color='red')
#plt.savefig("test.png")
plt.tight_layout()
plt.savefig("/home/alex/Dokumente/Bach/figures/heatwave.png", dpi=300)
plt.show()