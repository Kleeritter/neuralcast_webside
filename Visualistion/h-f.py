import matplotlib.pyplot as plt
import pylab as pl
import xarray as xr
import numpy as np
import pandas as pd

forecast_var = 'rain'
nc_path = '../Data/stunden/'+str(2022)+'_resample_stunden.nc' # Replace with the actual path to your NetCDF file
lstm_uni_path="forecast_lstm_uni.nc"
nhit_path="nhit.nc"
lstm_multi_path="forecast_lstm_multi.nc"
sarima_path="forecast_sarima.nc"


obs=xr.open_dataset(nc_path).to_dataframe()[forecast_var]
lstm_uni=xr.open_dataset(lstm_uni_path).to_dataframe()[forecast_var]
sarima=xr.open_dataset(sarima_path).to_dataframe()[forecast_var]
lstm_multi=xr.open_dataset(lstm_multi_path).to_dataframe()[forecast_var]
nhit=xr.open_dataset(nhit_path).to_dataframe()[forecast_var]
print(nhit.describe())
lstm_uni.loc[lstm_uni < 0.000518] = 0
sarima.loc[sarima < 0.000518] = 0

nhit.loc[nhit < 0.091275] = 0
lstm_multi.loc[lstm_multi < 0.001401] = 0

print(obs.describe())
print(sarima.describe())
models = [sarima,lstm_uni ,lstm_multi,nhit]
# Überprüfung auf Hits



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
fig, ax = plt.subplots()
def fsar(data):
    fsr=0
    for observed_value, calculated_value in zip(obs, data):
        if observed_value == 0 and calculated_value > 0:
            fsr += 1
    total_samples = len(obs)
    false_alarm_rate = (fsr / total_samples) #* 100

    print("FSR: {:.2f}%".format(false_alarm_rate))
    return false_alarm_rate

modelss=["(S)ARIMA", "LSTM","LSTM-Multi","NBeats"]
for i in range(len(models)):
    #print(model)
    x = []
    y = []
    y.append(hitc(models[i]))
    x.append(fsar(models[i]))
    ax.scatter(x, y,label=modelss[i])


#ax.axis([0, 1, 0, 1])
plt.ylabel("H=a/(a+c)")
plt.xlabel("F=b/(b+d)")
plt.title("H-F Diagramm")
ax.legend()
ax.set_yscale('log')
plt.set_cmap("magma")
ax.grid(True)

#ax.set_yticks([0,0., 1, ])
#pl.ylim=(0,1)#=[0,1]
#pl.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
#pl.yticks(np.arange(0,1,0.2))
#plt.scatter(x,y)
plt.show()