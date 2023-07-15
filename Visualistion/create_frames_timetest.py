import datetime
import xarray as xr
from Model.funcs.visualer_funcs import multilstm_full
import pandas as pd
import numpy as np

def forecast_lstm_multi(model_folder,params_folder,var_list,output_file,forecast_horizont,window_size):
    import glob
    #var_list = [
     #   "temp", "press_sl", "humid"]
    #var_list= [ "press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50",     "rain", "wind_10", "wind_50","wind_dir_50_sin", "temp"]
    forecast_year=2022
    dt = datetime.datetime(forecast_year, 1, 1, 0, 0)  # + datetime.timedelta(hours=window_size)
    dtl = datetime.datetime(forecast_year - 1, 12, 31, 23)
    dtlast = dtl - datetime.timedelta(hours=window_size )
    dtlast = dtl - datetime.timedelta(hours=window_size )

    nc_path = '../Data/stunden/' + str(
        forecast_year) + '_resample_stunden.nc'  # Replace with the actual path to your NetCDF file
    nc_path_last = '../Data/stunden/' + str(forecast_year - 1) + '_resample_stunden.nc'
    data = xr.open_dataset(nc_path)  # .to_dataframe()#["index">dt]
    datalast = xr.open_dataset(nc_path_last)
    # print(data.to_dataframe().iloc[-1])
    data = xr.concat([datalast, data], dim="index").to_dataframe()
    start_index_forecast = data.index.get_loc(dtlast)
    forecast_data = data[start_index_forecast:]
    start_index_visual = data.index.get_loc(dt)
    visual_data = data[start_index_visual:]
    #print(forecast_data.head())
    print((len(forecast_data.index.tolist()) )/forecast_horizont)

    data = {}
    for forecast_var in var_list:
        print(forecast_var)
        lost_window=0
        pred_lis = []
        multivariant_model_path = model_folder#+"/best_model_state_" + forecast_var +'_'+str(window_size)+'_'+str(forecast_horizont)+ ".pt"#../Model/opti/output/lstm_multi/models/best_model_state_" + forecast_var + ".pt"
        lstm_multi_params = params_folder#+'/best_params_lstm_multi_' + forecast_var + '.yaml'#'../Model/opti/output/lstm_multi/best_params_lstm_multi_' + forecast_var + '.yaml'
        print(window_size,forecast_horizont,len(forecast_data.index.tolist()))
        max_iterations = len(forecast_data) // forecast_horizont
        print(max_iterations)
        #for window, last_window in zip(range(window_size, len(forecast_data.index.tolist())-1, forecast_horizont),range(0, len(forecast_data.index.tolist()) - window_size,forecast_horizont)):

            #print(len(forecast_data[last_window:window]))
        for i in range(0,max_iterations-13):
            start_window = i * forecast_horizont
            end_window = start_window + window_size
            last_window = end_window - forecast_horizont
            #print(start_window,end_window,last_window)
            predictions_multi = multilstm_full(multivariant_model_path, forecast_data, start_idx=start_window,
                                               end_idx=end_window, forecast_var=forecast_var,hyper_params_path=lstm_multi_params,forecast_horizon=forecast_horizont,window_size=window_size)
            pred_lis.append(predictions_multi)
            lost_window += 1
            #lost_last_window = last_window
        data[forecast_var]=np.array(pred_lis).flatten()
    #print(window,last_window)
    df = pd.DataFrame(data)
    #if len(df>visual_data.index.tolist()
    #print(lost_window)
    #print(df)

    output_file = output_file#"forecast_lstm_multi.nc"
    df = df.set_index(pd.to_datetime(visual_data.index.tolist()), inplace=False)
    df.index.name = "Datum"
    df = xr.Dataset.from_dataframe(df).rename({forecast_var: str(window_size) +'_'+str(forecast_horizont)+'_'+forecast_var})
    df.to_netcdf(output_file)