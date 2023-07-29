import pandas as pd
from pmdarima import auto_arima
import xarray as xr
import random
from tqdm import tqdm
from collections import Counter
import yaml
random.seed(42)
nc_path = '../../Data/stunden/2022_resample_stunden.nc'#'../../Data/zusammengefasste_datei_2016-2019.nc' # Replace with the actual path to your NetCDF file

data = xr.open_dataset(nc_path).to_dataframe()

forecastvar ="wind_dir_50"
# Annahme: Ihre Zeitreihendaten sind im DataFrame df und die Zeitreihe ist in der Spalte "temp".
y = data[forecastvar]

seasonal=False

seasonal_period = 0

# Anzahl der Stunden in einem Block
block_size = 72

# Mindestabstand zwischen den Blöcken
min_block_distance = 24

# Anzahl der zufälligen Proben, die Sie erstellen möchten
num_samples = 15

maxiters = 40
# Liste zur Speicherung der besten Modelle für jeden Block
best_models = []

# Schleife, um zufällige Proben zu erstellen und das beste Modell für jede Probe zu finden
for _ in tqdm(range(num_samples)):
    # Zufälligen Startindex für den Block auswählen
    start_idx = random.randint(0, len(y) - block_size)

    # Auswählen des aktuellen Blocks
    y_block = y[start_idx : start_idx + block_size]

    # Verwenden von auto_arima, um die besten SARIMA-Parameter automatisch zu ermitteln
    model = auto_arima(y_block, seasonal=seasonal, m=seasonal_period, stepwise=True, trace=False, start_p=0, max_p=3, start_q=0, max_q=3,start_d=0,start_D=0,
                       start_P=0, max_P=3, start_Q=0, max_Q=3, max_D=3,max_d=3,maxiter=maxiters)

    # Die besten Parameter werden im 'model' Objekt gespeichert
    best_models.append((model.order, model.seasonal_order))
    print( model.order ,model.seasonal_order,start_idx)

# Berechnen des Durchschnitts der besten Modelle über alle Proben
parameter_counts = Counter(best_models)

# Finden der am häufigsten vorkommenden Parameter
most_common_non_seasonal_params = parameter_counts.most_common(1)[0][0][0]
most_common_seasonal_params = parameter_counts.most_common(1)[0][0][1]


print("Durchschnittliche nicht-saisonale Parameter (p, d, q):", most_common_non_seasonal_params)
print("Durchschnittliche saisonale Parameter (P, D, Q, s):", most_common_seasonal_params)



output_params = {
    'non_seasonal_params': {
        'p': most_common_non_seasonal_params[0],
        'd': most_common_non_seasonal_params[1],
        'q': most_common_non_seasonal_params[2]
    },
    'seasonal_params': {
        'P': most_common_seasonal_params[0],
        'D': most_common_seasonal_params[1],
        'Q': most_common_seasonal_params[2],
        's': most_common_seasonal_params[3]
    }
}


with open('bestparams/best_sarima_params_'+forecastvar+'.yaml', 'w') as file:
    yaml.dump(output_params, file)