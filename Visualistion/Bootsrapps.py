import xarray as xr
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def bootstrap_rmse(data1, data2, n_iterations=1000):
    np.random.seed(42)  # Für die Reproduzierbarkeit der Ergebnisse
    n_samples = len(data1)

    rmse_values = []
    for _ in range(n_iterations):
        sample_indices = np.random.randint(0, n_samples, n_samples)
        sample_data1 = data1.iloc[sample_indices]
        sample_data2 = data2.iloc[sample_indices]

        rmse = np.sqrt(mean_squared_error(sample_data1, sample_data2))
        rmse_values.append(rmse)

    return rmse_values

nc_path = '../Data/stunden/'+str(2022)+'_resample_stunden.nc' # Replace with the actual path to your NetCDF file
references_path="forecast_sarima.nc"
lstm_uni_path="forecast_lstm_uni.nc"
lstm_multi_path="time_test_better_a.nc"#"../Model/timetest/lstm_multi/output/temp/timetest_lstm_multitemp_24_24.nc"#"forecast_lstm_multi.nc"
tft_path="tft_dart.nc"
cors_path="cortest_all.nc"#"../Model/cortest/lstm_multi/output/temp/cortest_lstm_multitemp_24_24.nc"
nhits="nhit.nc"
prohet_path="prophet.nc"
lstm_uni=xr.open_dataset(lstm_uni_path).to_dataframe()
lstm_multi=xr.open_dataset(lstm_multi_path).to_dataframe()
LSTM_MULTI_CORS=xr.open_dataset(cors_path).to_dataframe()
tft=xr.open_dataset(tft_path).to_dataframe()
data = xr.open_dataset(nc_path).to_dataframe()
references= xr.open_dataset(references_path).to_dataframe()#[:-24]
nhits=xr.open_dataset(nhits).to_dataframe()
prophet=xr.open_dataset(prohet_path).to_dataframe()

target_parameter = 'humid'  # Ersetzen Sie 'parameter_name' durch den tatsächlichen Namen des Parameters

# Extrahieren Sie die Zeitreihen für den Zielparameter aus beiden DataFrames
data1_target = prophet[target_parameter]
data2_target = data[target_parameter]

# Berechnen Sie den RMSE mit dem Bootstrapping-Verfahren
rmse_values = bootstrap_rmse(data1_target, data2_target, n_iterations=1000)

# Zeigen Sie die Ergebnisse an, zum Beispiel den durchschnittlichen RMSE und die Standardabweichung
average_rmse = np.mean(rmse_values)
std_dev_rmse = np.std(rmse_values)
print(f"Durchschnittlicher RMSE: {average_rmse}")
print(f"Standardabweichung des RMSE: {std_dev_rmse}")


#plt.figure(figsize=(8, 6))
plt.grid(True)

plt.hist(rmse_values, bins='auto', edgecolor='k')
plt.xlabel('RMSE')
plt.ylabel('Häufigkeit')
plt.title('Histogramm der RMSE-Werte')

plt.show()