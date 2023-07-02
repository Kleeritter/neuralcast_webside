import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import xarray as xr
#QT_QPA_PLATFORM=wayland
r = np.arange(0, 2, 0.01)
theta = 2 * np.pi * r
from sklearn.preprocessing import MinMaxScaler

#observations = np.array([1, 2, 3, 4, 5])
#model1 = np.array([0.8, 1.9, 2.7, 3.8, 5.2])
#model2 = np.array([1, 2, 3, 4, 5])
#forecast_var = 'wind_dir_50_cos'
var_list = ["wind_dir_50_sin", "wind_dir_50_cos", 'temp']
nc_path = '../Data/stunden/'+str(2022)+'_resample_stunden.nc' # Replace with the actual path to your NetCDF file

references="forecast_sarima.nc"
lstm_uni_path="forecast_lstm_uni.nc"
lstm_multi_path="forecast_lstm_multi.nc"
tft_path="forecast_tft.nc"


markers= ["o", "v", "s", "p", "P", "*", "X", "D", "d", "1", "2", "3", "4", "8", "h", "H", "+", "x", "X", "|", "_"]
rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
tlocs = np.arcsin(rlocs)
srange=(0, 1.5)
tmax = np.pi/2
smin = srange[0] #* np.std(observations)
smax = srange[1] #*np.std(observations)
rs, ts = np.meshgrid(np.linspace(smin, smax),
                     np.linspace(0, tmax))
# Compute centered RMS difference
rms = np.sqrt(1 ** 2 + rs ** 2 - 2 * 1 * rs * np.sin(ts))

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
def tayl(forecast_var,marker):
    trad = xr.open_dataset(references).to_dataframe()[forecast_var]
    lstm_uni = xr.open_dataset(lstm_uni_path).to_dataframe()[forecast_var]
    lstm_multi = xr.open_dataset(lstm_multi_path).to_dataframe()
    tft = xr.open_dataset(tft_path).to_dataframe()
    observations = xr.open_dataset(nc_path).to_dataframe()[forecast_var]
    # Berechnung der Korrelationskoeffizienten
    correlation1 = np.arcsin(np.corrcoef(observations, trad)[0, 1])
    correlation2 = np.arcsin(np.corrcoef(observations, lstm_uni)[0, 1])
    stdv1 = np.array(np.std(trad))
    stdv2 = np.array(np.std(lstm_uni))

    #print(np.corrcoef(observations, model2))
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = np.array([0, np.std(observations)])
    ref = scaler.fit_transform(values.reshape(-1, 1))
    stdv1 = scaler.transform(stdv1.reshape(-1, 1)).flatten()
    stdv2 = scaler.transform(stdv2.reshape(-1, 1)).flatten()




    ax.scatter(correlation1, stdv1, color='blue', marker=marker,edgecolors='black')
    ax.scatter(correlation2, stdv2,color="red", marker=marker)

    ax.set_rmax(2)
    #ax.set_thetaticks([1,0])
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rmax(1.5)
    ax.set_rticks([0.5, 1, ])  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)
    ax.set_xticks(tlocs)
    ax.set_xticklabels(rlocs)
    if i==len(var_list)-1:

        contours = ax.contour(ts, rs, rms, levels=5, colors='blue', linewidths=1, linestyles='solid')
        ax.clabel(contours, inline=True, fontsize=10)
        ax.annotate('Observed',
                    xy=(np.pi / 2, 1),  # theta, radius
                    xytext=(np.pi / 2.1, 1.15),  # fraction, fraction
                    arrowprops=dict(facecolor='black', shrink=0.005, headwidth=5, width=1, headlength=5),
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    )
    elif i==0:
        ax.plot(theta, np.ones(200), color='black', linestyle='dashed')

    return

for i in range(0, len(var_list)):
    tayl(var_list[i], markers[i])

#ax.clabel(contours, inline=True, fontsize=10)
ax.set_title("Taylor Diagramm", va='bottom')
var_handles = []
var_labels = []

model_handles = []
model_labels = []

for i in range(len(var_list)):
    var_handle = ax.scatter([], [], color='blue', marker=markers[i], edgecolor='black')
    var_handles.append(var_handle)
    var_labels.append(var_list[i])

    model_handle = ax.scatter([], [], color='blue', marker='o', edgecolor='black')  # Hier die gewünschten Farben für die Modelle einsetzen
    model_handles.append(model_handle)
    model_labels.append('Modell ' + str(i+1))  # Hier die gewünschten Labels für die Modelle einsetzen

ax.plot([], [], color='black', linestyle='dashed')

var_legend = ax.legend(var_handles, var_labels, loc='upper left', title='Variablen')
model_legend = ax.legend(model_handles, model_labels, loc='upper right', title='Modelle')

# Füge beide Legenden zusammen, damit sie gemeinsam angezeigt werden
ax.add_artist(var_legend)
ax.add_artist(model_legend)
plt.show()