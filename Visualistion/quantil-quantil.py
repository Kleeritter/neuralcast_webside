import matplotlib.pyplot as plt
import pylab as pl
import xarray as xr
import numpy as np
import pandas as pd
import statsmodels.api as sm

forecast_var = 'temp'
nc_path = '../Data/stunden/'+str(2022)+'_resample_stunden.nc' # Replace with the actual path to your NetCDF file
lstm_uni_path="forecast_lstm_uni.nc"
sarima_path="forecast_sarima.nc"
lstm_multi_path="forecast_lstm_multi.nc"
nhits_path="nhit.nc"

obs=xr.open_dataset(nc_path).to_dataframe()[forecast_var]
lstm_uni=xr.open_dataset(lstm_uni_path).to_dataframe()[forecast_var]
sarima=xr.open_dataset(sarima_path).to_dataframe()[forecast_var]
lstm_multi=xr.open_dataset(lstm_multi_path).to_dataframe()[forecast_var]
nhits=xr.open_dataset(nhits_path).to_dataframe()[forecast_var]

modellist=[lstm_uni,lstm_multi,sarima,nhits]
fig, axs = plt.subplots(2,2)#, figsize=(10, 10))
ax1 = axs[0, 0]
ax2 = axs[0, 1]
ax3 = axs[1, 0]
ax4 = axs[1, 1]
axes=[ax1, ax2,ax3,ax4]

figs,axe=plt.subplots()
def qq(modeln,ax, modelname):
    ax.scatter(modeln, obs,label='Observation',color='lightsteelblue')
    ax.set_title(modelname)
    quantiles = [0.10,0.25, 0.5, 0.75,0.9]  # Wähle die gewünschten Quantile
    combined_df = pd.DataFrame({
            'lstm_uni': np.array(modeln.values).flatten(),
            'obs': np.array(obs.values).flatten()

        })
    models=[]
    for q in quantiles:
        model = sm.QuantReg(combined_df['lstm_uni'], sm.add_constant(combined_df['obs'])).fit(q=q)
        models.append(model)
    for i, q in enumerate(quantiles):
        y_quantile = models[i].predict(sm.add_constant(combined_df['obs']))
        ax.plot(obs, y_quantile, label=f'{int(q*100)}th Quantile')
    plt.axis([260, 315, 260, 315])
    axst = ax.twinx()
    bins=np.arange(260, 315, 1)
    axst.axis([260, 315, 0, 2800])
    axst.set_yticks([0,100,300,400, 500])
    yticks = range(0, 801, 200)
    axst.hist(modeln, bins=bins,align="mid", density=False, color='lightsteelblue',rwidth=0.2, label='Frequency')
    plt.ylabel("Samples",loc='bottom')


#for ax, hist_ax,model in zip(axes, axs.flatten(),modellist):
 #   hist_ax.yaxis.tick_right()
  #  qq(model, ax, hist_ax)

modelname=["sarima","lstm_uni","lstm_multi","nhits"]
qq(sarima,ax1,modelname[0])
qq(lstm_uni,ax2,modelname[1])
qq(lstm_multi,ax3,modelname[2])
qq(nhits,ax4,modelname[3])

for ax in axs.flat:
    ax.set(xlabel='gemessene Temperatur in K', ylabel='simulierte Temperatur in K')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.tight_layout()
plt.show()