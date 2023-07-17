import numpy as np
import matplotlib
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import math
from sklearn.metrics import mean_squared_error
import seaborn as sns

def rmses_month(model,var,data,modelvar="Null"):
    list=[]
    for month in range(1,13,1):
        if modelvar !="Null":
            list.append(math.sqrt(mean_squared_error(data.loc[(data.index.month == month)][var],
                                                     model.loc[(model.index.month == month)][modelvar])))
        else:
            list.append(math.sqrt(mean_squared_error(data.loc[(data.index.month==month)][var], model.loc[(model.index.month==month)][var])))
    return list

forecast_year=2022

nc_path = '../Data/stunden/'+str(forecast_year)+'_resample_stunden.nc' # Replace with the actual path to your NetCDF file

#references=np.load("sarima/reference_temp_.npy").flatten()
references_path="forecast_sarima.nc"
lstm_uni_path="forecast_lstm_uni.nc"
lstm_multi_path="../Model/timetest/lstm_multi/output/temp/timetest_lstm_multitemp_24_24.nc"#"forecast_lstm_multi.nc"
tft_path="tft_dart.nc"

nhits="nhit.nc"
lstm_uni=xr.open_dataset(lstm_uni_path).to_dataframe()
lstm_multi=xr.open_dataset(lstm_multi_path).to_dataframe()
tft=xr.open_dataset(tft_path).to_dataframe()
data = xr.open_dataset(nc_path).to_dataframe()
references= xr.open_dataset(references_path).to_dataframe()
nhits=xr.open_dataset(nhits).to_dataframe()

lstm_multi_org="forecast_lstm_multi.nc"
lstm_multi_org=xr.open_dataset(lstm_multi_org).to_dataframe()
vars=["temp","humid","press_sl","diffuscmp11"]
fig, ax = plt.subplots(2,2,figsize=(15, 10))
sns.set_theme(style="darkgrid")
def monthsdataf(var,data,modelvar="Null",multidata=lstm_multi_org):
    months=pd.DataFrame({
            'Month': np.arange(1,13,1),#["Jan","Feb","Mär","April","Mai",],
            'RMSE_multi':rmses_month(multidata,var,data,modelvar=modelvar),#"24_24_temp"),
            'RMSE_uni':rmses_month(lstm_uni,var,data),
            'RMSE_nhits': rmses_month(nhits, var, data),
            'RMSE_sarima':rmses_month(references,var,data),
        })
    return months

sns.lineplot(x="Month", y='value', hue='variable',data=pd.melt(monthsdataf(var=vars[0],data=data,modelvar="24_24_"+vars[0],multidata=lstm_multi), ['Month']),ax=ax[0,0])
sns.lineplot(x="Month", y='value', hue='variable',data=pd.melt(monthsdataf(var=vars[1],data=data), ['Month']),ax=ax[1,0])
sns.lineplot(x="Month", y='value', hue='variable',data=pd.melt(monthsdataf(var=vars[2],data=data), ['Month']),ax=ax[0,1])#
sns.lineplot(x="Month", y='value', hue='variable',data=pd.melt(monthsdataf(var=vars[3],data=data), ['Month']),ax=ax[1,1])
plt.show()
def monhtly_power(data):
    data=data.groupby(data.index.month).sum()
    return data