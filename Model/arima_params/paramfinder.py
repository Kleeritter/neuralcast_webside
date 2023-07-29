import pandas as pd
from pmdarima import auto_arima
import xarray as xr

nc_path = '../../Data/zusammengefasste_datei_2016-2019.nc' # Replace with the actual path to your NetCDF file

data = xr.open_dataset(nc_path).to_dataframe()

# Annahme: Ihre Zeitreihendaten sind im DataFrame df und die Zeitreihe ist in der Spalte "value".
y = data['temp'][-672:]

# Verwenden Sie auto_arima, um die besten SARIMA-Parameter automatisch zu ermitteln.
model = auto_arima(y, seasonal=True, m=24, stepwise=True, trace=True, start_p=1, max_p=3, start_q=1, max_q=3,
                   start_P=1, max_P=3, start_Q=1, max_Q=3)

# Die besten Parameter werden im 'model' Objekt gespeichert.
print(model.order)  # Nicht-saisonale Parameter (p, d, q)
print(model.seasonal_order)  # Saisonale Parameter (P, D, Q, s)
