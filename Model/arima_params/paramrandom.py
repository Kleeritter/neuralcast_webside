# Import necessary libraries
import pandas as pd
from pmdarima import auto_arima
import xarray as xr
import random
from tqdm import tqdm
from collections import Counter
import yaml

# Set a random seed for reproducibility
random.seed(42)

# Define the path to the NetCDF file
nc_path = '../../Data/stunden/2022_resample_stunden.nc'

# Load the data from the NetCDF file into a DataFrame
data = xr.open_dataset(nc_path).to_dataframe()

# Define a function to find the best SARIMA parameters for forecasting
def paramfinder(forecast_var, seasonal=True):
    # Define the variable to forecast
    forecastvar = forecast_var

    # Extract the time series data
    y = data[forecastvar]

    # Set parameters for SARIMA
    seasonal = seasonal
    seasonal_period = 24
    block_size = 72
    min_block_distance = 24
    num_samples = 30
    maxiters = 40
    best_models = []

    # Iterate through samples to find the best model parameters
    for _ in tqdm(range(num_samples)):
        start_idx = random.randint(0, len(y) - block_size)
        y_block = y[start_idx: start_idx + block_size]

        # Use auto_arima to find the best SARIMA parameters
        model = auto_arima(y_block, seasonal=seasonal, m=seasonal_period, stepwise=True, trace=False,
                           start_p=0, max_p=3, start_q=0, max_q=3, start_d=0, start_D=0,
                           start_P=0, max_P=3, start_Q=0, max_Q=3, max_D=3, max_d=3, maxiter=maxiters)

        best_models.append((model.order, model.seasonal_order))
        print(model.order, model.seasonal_order, start_idx)

    # Calculate the average of the best models over all samples
    parameter_counts = Counter(best_models)
    most_common_non_seasonal_params = parameter_counts.most_common(1)[0][0][0]
    most_common_seasonal_params = parameter_counts.most_common(1)[0][0][1]

    # Print and save the most common parameters
    print("Average non-seasonal parameters (p, d, q):", most_common_non_seasonal_params)
    print("Average seasonal parameters (P, D, Q, s):", most_common_seasonal_params)

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

    # Save the best parameters in a YAML file
    with open('bestparams/best_sarima_params_' + forecastvar + '.yaml', 'w') as file:
        yaml.dump(output_params, file)
    return

# Define a list of variables to forecast
forecast_vars = ["press_sl"]

# Iterate through the forecast variables and find their best SARIMA parameters
for forecast_var in forecast_vars:
    paramfinder(forecast_var)
