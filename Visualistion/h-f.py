import matplotlib.pyplot as plt
import pylab as pl
import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.dates as mdates


forecast_var = 'rain'
nc_path = '../Data/stunden/'+str(2022)+'_resample_stunden.nc' # Replace with the actual path to your NetCDF file
lstm_uni_path= "time_test_single.nc"#"forecast_lstm_uni.nc"
nhit_path="../Model/baseline/baseline.nc"
lstm_multi_path="time_test_better_a.nc"
lstm_multi_cor_path="cortest_all.nc"
sarima_path="auto_arima.nc"


obs=xr.open_dataset(nc_path).to_dataframe()[forecast_var]
lstm_uni=xr.open_dataset(lstm_uni_path).to_dataframe()[forecast_var]
sarima=xr.open_dataset(sarima_path).to_dataframe()[forecast_var]
lstm_multi=xr.open_dataset(lstm_multi_path).to_dataframe()[forecast_var]
baseline=xr.open_dataset(nhit_path).to_dataframe()[forecast_var]
lstm_multi_cor=xr.open_dataset(lstm_multi_cor_path).to_dataframe()[forecast_var]

daystart=pd.to_datetime(str(2022)+"-12-25 00:00")
dayend=pd.to_datetime(str(2022)+"-12-30 23:00")
models = [sarima,lstm_uni ,lstm_multi,baseline,lstm_multi_cor]




fsr=0
def hitc(data):
    hits = 0
    for observed_value, calculated_value in zip(obs, data):
        if observed_value > 0 and calculated_value > 0:
            hits += 1

    # Hitrate berechnen
    total_samples = len(obs)
    hitrate = (hits / total_samples)# * 100

    print("Hitrate: {:.2f}%".format(hitrate))
    return hitrate
fig, ax = plt.subplots(ncols=2)
#sns.set_context("paper", font_scale=1.5)
def fsar(data):
    fsr=0
    for observed_value, calculated_value in zip(obs, data):
        if observed_value == 0 and calculated_value > 0:
            fsr += 1
    total_samples = len(obs)
    false_alarm_rate = (fsr / total_samples) #* 100

    print("FSR: {:.2f}%".format(false_alarm_rate))
    return false_alarm_rate

modelss=["(S)ARIMA", "LSTM","LSTM-Multi","baseline","LSTN-Multi-Cor"]
colorlist=["red","blue","green","orange","purple","black"]
for i in range(len(models)):
    #print(model)
    x = []
    y = []
    y.append(hitc(models[i]))
    x.append(fsar(models[i]))
    sns.scatterplot(x=x, y=y,label=modelss[i],ax=ax[0],s=100,color=colorlist[i])

def frame(var,cor=0):
    winds = pd.DataFrame({
        'Date': obs.loc[daystart:dayend].index,
        'SARIMA': 100*sarima.loc[daystart:dayend],
        'univariate LSTM': 100* lstm_uni.loc[daystart:dayend],
        'multivariate LSTM': 100*lstm_multi.loc[daystart:dayend]-cor,


        'baseline':100*baseline.loc[daystart:dayend],
        'cor. multi. LSTM':100*lstm_multi_cor.loc[daystart:dayend],
        'observed': 100* obs.loc[daystart:dayend] - cor
    })
    return winds
#ax.axis([0, 1, 0, 1])
ax[0].set_ylabel("H=a/(a+c)")
ax[0].set_xlabel("F=b/(b+d)")
ax[0].set_title("H-F diagram")
ax[0].legend()
ax[0].set_yscale('log')

ax[0].grid(True)
sns.lineplot(x="Date", y='value', hue='variable',data=pd.melt(frame(var="rain",cor=0), ['Date']),ax=ax[1],legend=False,palette=colorlist)

ax2 = ax[1].twinx()
ax[1].get_shared_y_axes().join(ax[1], ax2)
ax[1].set_ylabel("")
ax[1].set_title("Precipitation in December 2022")


ax[1].tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)

ax2.set_ylabel("Precipitation [mm]")
date_format = mdates.DateFormatter('%d.')
ax[1].xaxis.set_major_formatter(date_format)


years_locator = mdates.DayLocator(interval=1)
ax[1].xaxis.set_major_locator(years_locator)
#plt.grid(True)
plt.tight_layout()
plt.savefig("/home/alex/Dokumente/Bach/figures/raino.png", dpi=300, bbox_inches="tight")

plt.show()