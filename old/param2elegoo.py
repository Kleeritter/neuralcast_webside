import pandas as pd
import numpy as np
import pmdarima as pm
from sklearn.model_selection import TimeSeriesSplit
import xarray as xr

nc_path = '../../Data/stunden/'+str(2022)+'_resample_stunden.nc' # Replace with the actual path to your NetCDF file
array=xr.open_dataset(nc_path)
time_series_data = array.to_dataframe()["temp"]
array.close()
# Annahme: Ihre Zeitreihe ist in der Variablen "time_series_data" gespeichert
# Die Zeitreihe in ein pandas DataFrame konvertieren (falls sie es nicht bereits ist)
data = pd.DataFrame({'timestamp': pd.date_range(start='2022-01-01', periods=len(time_series_data), freq='H'),
                     'value': time_series_data})

# Zeitreihenindex setzen
data.set_index('timestamp', inplace=True)

# Anzahl der Werte in einem Schiebefenster
window_size = 672

# Anzahl der Schritte, um das Schiebefenster zu verschieben (24 Stunden)
step_size = 24

# Funktion zur Berechnung der SARIMA-Parameter für eine Charge
def find_sarima_params(data_chunk):
    model = pm.auto_arima(data_chunk['value'],
                          seasonal=True,
                          m=24,
                          stepwise=True,
                          suppress_warnings=True,
                          error_action="ignore",
                          trace=False)
    return model.order, model.seasonal_order, model.aic()

# Batch-Verarbeitung
tscv = TimeSeriesSplit(n_splits=len(data)//window_size)
best_order, best_seasonal_order = None, None
best_aic = float("inf")

for train_index, test_index in tscv.split(data):
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]

    # Hier wird die SARIMA-Parameter-Berechnung für jede Charge durchgeführt
    order, seasonal_order, aic = find_sarima_params(train_data)

    if aic < best_aic:
        best_aic = aic
        best_order = order
        best_seasonal_order = seasonal_order

print(f"Best SARIMA Order: {best_order}")
print(f"Best SARIMA Seasonal Order: {best_seasonal_order}")
print(f"Best AIC: {best_aic}")