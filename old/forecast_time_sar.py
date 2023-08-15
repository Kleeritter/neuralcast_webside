import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import xarray as xr
import matplotlib.pyplot as plt
nc_path = '../Data/stunden/'+str(2022)+'_resample_stunden.nc' # Replace with the actual path to your NetCDF file
references_path="auto_arima.nc"
lstm_uni_path="forecast_lstm_uni.nc"
lstm_multi_path="time_test_better_a.nc"#"../Model/timetest/lstm_multi/output/temp/timetest_lstm_multitemp_24_24.nc"#"forecast_lstm_multi.nc"
tft_path="tft_dart.nc"
cors_path="cortest_all.nc"#"../Model/cortest/lstm_multi/output/temp/cortest_lstm_multitemp_24_24.nc"
nhits="nhit.nc"
prohet_path="prophet.nc"

ruthe_path="../Data/ruthe_2020_resample_stunden.nc"
ruthe=xr.open_dataset(ruthe_path).to_dataframe()
ruthe_forecast="ruthe_forecast.nc"
ruthe_forecast=xr.open_dataset(ruthe_forecast).to_dataframe()
lstm_uni=xr.open_dataset(lstm_uni_path).to_dataframe()
lstm_multi=xr.open_dataset(lstm_multi_path).to_dataframe()
LSTM_MULTI_CORS=xr.open_dataset(cors_path).to_dataframe()
tft=xr.open_dataset(tft_path).to_dataframe()
data = xr.open_dataset(nc_path).to_dataframe()
references= xr.open_dataset(references_path).to_dataframe()#[:-24]
nhits=xr.open_dataset(nhits).to_dataframe()
prophet=xr.open_dataset(prohet_path).to_dataframe()

def timeshit(time):
    # Angenommen, du hast bereits die beiden DataFrames df1 und df2

    # Zuerst die DataFrames so vorbereiten, dass sie nur die ersten zwei Stunden jedes Tages enthalten
    df1_first_two_hours = references[references.index.hour < time]
    df2_first_two_hours = data[data.index.hour < time]

    # Überprüfen, ob die beiden DataFrames für dieselben Tage Daten haben
    common_dates = df1_first_two_hours.index.intersection(df2_first_two_hours.index)

    # Nur die gemeinsamen Daten für den RMSE berücksichtigen
    df1_common = df1_first_two_hours.loc[common_dates]
    df2_common = df2_first_two_hours.loc[common_dates]

    # Den RMSE berechnen
    rmse = np.sqrt(mean_squared_error(df1_common["temp"], df2_common["temp"]))
    return rmse

def timeshit2(time ,model=references):
    # Erstelle eine Liste mit den Zeitindizes für die ersten beiden Stunden jedes Tages
    starts=[i for i in range(1,time+1)]
    #time_indices = starts + [[1 + 24 * i for i in range(365)]]
    time_indices=[]
    for start in starts:
        time_indices.append( [start + 24 * i for i in range(365)])

    time_indices = np.array(time_indices).flatten()
    #print(time_indices)

    # Extrahiere die relevanten Zeilen aus den DataFrames basierend auf den Zeitindizes
    df1_selected = model.iloc[time_indices]
    df2_selected = data.iloc[time_indices]

    # Überprüfe, ob die beiden DataFrames Daten für die gleichen Zeitindizes haben
    #common_indices = df1_selected.index.intersection(df2_selected.index)

    # Extrahiere nur die gemeinsamen Daten für den RMSE
   # df1_common = df1_selected.iloc[time_indices]
    #df2_common = df2_selected.iloc[time_indices]
    return np.sqrt(mean_squared_error(df1_selected["temp"], df2_selected["temp"]))

rmsess=[]
rmses=[]
for i in range(1,24):
    rmses.append(timeshit2(i,model=references))
    rmsess.append(timeshit2(i,model=lstm_multi))

#print("Root Mean Squared Error (RMSE):", rmse)
plt.plot(rmses)
plt.plot(rmsess)
plt.show()