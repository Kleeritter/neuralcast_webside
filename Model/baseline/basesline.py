# Import necessary libraries

import xarray as xr
from tqdm import tqdm
import pandas as pd
from darts import TimeSeries, concatenate
from darts.models import NaiveSeasonal
import datetime
from multiprocessing import Pool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define forecast variables
forecast_vars = ["temp", "press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50", "rain", "wind_10",
                 "wind_50", "wind_dir_50_sin", "wind_dir_50_cos"]


# Define a function for DARIMA forecasting
def base(number):
    # Get the forecast variable based on the number
    forecast_var = forecast_vars[number]

    # Load data from the NetCDF file
    data = xr.open_dataset('../../Data/zusammengefasste_datei_2016-2022.nc').to_dataframe()[[forecast_var]]
    series = TimeSeries.from_dataframe(data, value_cols=forecast_var, freq="h")
    train, series = series.split_before(pd.Timestamp("2020-01-01 00:00:00"))

    # Set forecast horizon and window size
    forecast_horizont = 24
    window_size = 672

    # Determine start day for prediction
    startday = datetime.datetime(2022, 1, 1, 0) - datetime.timedelta(hours=window_size)
    rest, year_values = series.split_after(pd.Timestamp(startday))

    # Iterate through windows and perform forecasting
    for window, last_window in tqdm(zip(range(window_size, len(year_values), forecast_horizont),
                                        range(0, len(year_values) - window_size, forecast_horizont))):
        # Define the forecasting model (NaiveSeasonal)
        model = NaiveSeasonal(K=24)
        model.fit(year_values[last_window:window])

        # Predict for the current window
        if last_window == 0:
            pred_tmp = model.predict(n=forecast_horizont, verbose=False)
            df = pred_tmp.pd_dataframe()
        else:
            pres = model.predict(n=forecast_horizont, verbose=False)
            fs = pres.pd_dataframe()
            df = pd.concat([df, fs])

    # Truncate the predictions to 8760 points (1 year)
    if len(df) > 8760:
        df = df[:-(len(df) - 8760)]

    # Convert the predictions to a TimeSeries and save to a NetCDF file
    preds = TimeSeries.from_dataframe(df)
    output = preds.pd_dataframe()
    print(output.head())
    file = "output/" + forecast_var + str(window_size) + "_" + str(forecast_horizont) + ".nc"
    output = output.to_xarray().to_netcdf(file)


# Create a list of numbers corresponding to forecast variables
numbers = range(0, len(forecast_vars))

# Use multiprocessing to perform BASE forecasting for each forecast variable
with Pool(6) as p:
    p.map(base, numbers)
