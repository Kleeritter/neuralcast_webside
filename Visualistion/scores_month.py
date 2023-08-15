import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime
import seaborn as sns
import numpy as np
import math
from sklearn.metrics import mean_squared_error,mean_absolute_error


forecast_var="temp" #Which variable to forecast
window_size=24*7*4 #How big is the window for training
forecast_horizon=24 #How long to cast in the future
forecast_year=2022 #Which year to forecast
dt = datetime.datetime(forecast_year,1,1,0,0) #+ datetime.timedelta(hours=window_size)



nc_path = '../Data/stunden/'+str(forecast_year)+'_resample_stunden.nc' # Replace with the actual path to your NetCDF file

#references=np.load("sarima/reference_temp_.npy").flatten()
references_path="forecast_sarima.nc"
lstm_uni_path="forecast_lstm_uni.nc"
lstm_multi_path="time_test_better_a.nc"
lstm_multi_cor_path="cortest_all.nc"
tft_path="tft_dart.nc"
baseline_path="../Model/baseline/baseline.nc"

nhits="nhit.nc"
lstm_uni=xr.open_dataset(lstm_uni_path).to_dataframe()
lstm_multi=xr.open_dataset(lstm_multi_path).to_dataframe()
tft=xr.open_dataset(tft_path).to_dataframe()
data = xr.open_dataset(nc_path).to_dataframe()
references= xr.open_dataset(references_path).to_dataframe()
nhits=xr.open_dataset(nhits).to_dataframe()
baseline=xr.open_dataset(baseline_path).to_dataframe()
lstm_multi_cor=xr.open_dataset(lstm_multi_cor_path).to_dataframe()


def rmses_month(model,var):
    list=[]
    for month in range(1,13,1):
        #list.append(math.sqrt(mean_squared_error(data.loc[(data.index.month==month)][var], model.loc[(model.index.month==month)][var])))
        list.append((mean_absolute_error(data.loc[(data.index.month==month)][var], model.loc[(data.index.month==month)][var]))/(mean_absolute_error(data.loc[(data.index.month==month)][var],baseline.loc[(data.index.month==month)][var])))
    return list

def month_plot(var):
    referencess=rmses_month(references,var)
    months=pd.DataFrame({
        'Monht': np.arange(1,13,1),#["Jan","Feb","Mär","April","Mai",],
        #'RMSE_temp_multi':[x / np.max(lstm_multi.loc[data.index.month == z]) for x, z in zip(rmses_month(lstm_multi, var), np.arange(0, 13, 1))],
        #'RMSE_temp_sarima': [x / np.max(references.loc[data.index.month == z]) for x, z in
         #                   zip(rmses_month(references, var), np.arange(0, 13, 1))],
        #'RMSE_temp_uni': [x / np.max(lstm_uni.loc[data.index.month == z]) for x, z in
          #                  zip(rmses_month(lstm_uni, var), np.arange(0, 13, 1))],
        #'RMSE_press_multi': [x / np.max(lstm_multi.loc[data.index.month == z]) for x, z in
         #                   zip(rmses_month(lstm_multi, "press_sl"), np.arange(0, 13, 1))],
        'RMSE_temp_multi': rmses_month(lstm_multi, var),
        'RMSE_temp_sarima': rmses_month(references, var),
        'RMSE_temp_uni': rmses_month(lstm_uni, var),
        'RMSE_temp_multi_cor': rmses_month(lstm_multi_cor, var),
    })


    #sns.set_theme(style="darkgrid")
    sns.set_context("paper", font_scale=1.5)
    sns.lineplot(data=months,x="Monht", y="RMSE_temp_multi",label="LSTM Multi")
    sns.scatterplot(data=months, x="Monht", y="RMSE_temp_multi")
    sns.lineplot(data=months, x="Monht", y="RMSE_temp_sarima", label="SARIMA")
    sns.scatterplot(data=months, x="Monht", y="RMSE_temp_sarima")
    sns.lineplot(data=months, x="Monht", y="RMSE_temp_uni", label="LSTM Uni")
    sns.scatterplot(data=months, x="Monht", y="RMSE_temp_uni")
    sns.lineplot(data=months, x="Monht", y="RMSE_temp_multi_cor", label="LSTM Multi Cor")
    sns.scatterplot(data=months, x="Monht", y="RMSE_temp_multi_cor")
    plt.xlabel("Month")
    plt.ylabel("MASE")
    plt.title("MASE for temperature throughout the year")
    #fig.tight_layout()

    plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.grid(True)
    plt.savefig("/home/alex/Dokumente/Bach/figures/monthly_mase.png", dpi=300)


    #sns.lineplot(data=months, x="Monht", y="RMSE_press_multi", label="LSTM Multi Press")

    plt.show()
    #print(months["skills"])
    return

month_plot(forecast_var)