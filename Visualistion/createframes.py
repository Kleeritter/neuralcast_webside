from Model.funcs.visualer_funcs import lstm_uni,multilstm_full
import pandas as pd
import xarray as xr
import datetime
import numpy as np


forecast_var="temp" #Which variable to forecast
window_size=24*7*4 #How big is the window for training
forecast_horizon=24 #How long to cast in the future
forecast_year=2022 #Which year to forecast
dt = datetime.datetime(forecast_year,1,1,0,0) #+ datetime.timedelta(hours=window_size)
dtl=datetime.datetime(forecast_year -1 ,12,31,23)
dtlast= dtl - datetime.timedelta(hours=window_size-1)
nc_path = '../Data/stunden/'+str(forecast_year)+'_resample_stunden.nc' # Replace with the actual path to your NetCDF file
nc_path_last = '../Data/stunden/'+str(forecast_year-1)+'_resample_stunden.nc'
data = xr.open_dataset(nc_path)#.to_dataframe()#["index">dt]
datalast= xr.open_dataset(nc_path_last)

data=xr.concat([datalast,data],dim="index").to_dataframe()
start_index_forecast = data.index.get_loc(dtlast)
start_index_visual = data.index.get_loc(dt)
forecast_data=data[start_index_forecast:]

visual_data=data[start_index_visual:]



def forecast_lstm_uni(model_folder,params_folder,var_list,output_file,forecast_horizont,window_size,forecast_year=2022,ruthe=False):
    forecast_year=forecast_year
    dtl = datetime.datetime(forecast_year - 1, 12, 31, 23)
    dtlast = dtl - datetime.timedelta(hours=window_size - 1)
    dtlast = dtl - datetime.timedelta(hours=window_size - 1)
    if ruthe:
       nc_path = '../Data/ruthe_' + str(
            forecast_year)+'_resample_stunden.nc'
       nc_path_last = '../Data/ruthe_' + str(
            forecast_year-1)+'_resample_stunden.nc'
    else:
        nc_path = '../Data/stunden/' + str(
            forecast_year) + '_resample_stunden.nc'  # Replace with the actual path to your NetCDF file
        nc_path_last = '../Data/stunden/' + str(forecast_year - 1) + '_resample_stunden.nc'
    data = xr.open_dataset(nc_path)  # .to_dataframe()#["index">dt]
    datalast = xr.open_dataset(nc_path_last)
    # print(data.to_dataframe().iloc[-1])
    data = xr.concat([datalast, data], dim="index").to_dataframe()
    start_index_forecast = data.index.get_loc(dtlast)
    forecast_data = data[start_index_forecast:]
    #print(forecast_data.head())
    print((len(forecast_data.index.tolist()) - window_size)/forecast_horizont)

    data = {}


    for forecast_var in var_list:
        print(forecast_var)
        lost_window=0
        pred_lis = []
        univariant_model_path = model_folder#+"/best_model_state_" + forecast_var +'_'+str(window_size)+'_'+str(forecast_horizont)+ ".pt"#../Model/opti/output/lstm_multi/models/best_model_state_" + forecast_var + ".pt"
        lstm_uni_params = params_folder#+'/best_params_lstm_multi_' + forecast_var + '.yaml'#'../Model/opti/output/lstm_multi/best_params_lstm_multi_' + forecast_var + '.yaml'
        for window, last_window in zip(range(window_size, len(forecast_data.index.tolist()), forecast_horizont),range(0, len(forecast_data.index.tolist()) - window_size,forecast_horizont)):
            #print(len(forecast_data[last_window:window]))
           # predictions_multi = multilstm_full(multivariant_model_path, forecast_data, start_idx=last_window,
            #                                   end_idx=window, forecast_var=forecast_var,hyper_params_path=lstm_multi_params,forecast_horizon=forecast_horizont,window_size=window_size,cor_vars=corvars,numvars=len(corvars))
            predictions = lstm_uni(univariant_model_path, forecast_data[forecast_var], start_index=last_window,
                                   end_index=window,hyper_params_path=lstm_uni_params,forecast_var=forecast_var)  # .insert(0, Messfrühling[0:24]),

            pred_lis.append(predictions)
            lost_window += 1
            #lost_last_window = last_window
        data[forecast_var]=np.array(pred_lis).flatten()



    df = pd.DataFrame(data)
    if len(df)>len(visual_data):

        df=df[:-(len(df)-len(visual_data))]
    #print(lost_window)
    #print(df)
    output_file = output_file#"forecast_lstm_multi.nc"
    df = df.set_index(pd.to_datetime(visual_data.index.tolist()), inplace=False)
    df.index.name = "Datum"
    #df = xr.Dataset.from_dataframe(df).rename({forecast_var: str(window_size) +'_'+str(forecast_horizont)+'_'+forecast_var})
    df.to_xarray().to_netcdf(output_file)




def forecast_lstm_multi(model_folder,params_folder,var_list,output_file,forecast_horizont,window_size,corvars=[ "temp","press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50","rain", "wind_10", "wind_50","wind_dir_50_sin", "wind_dir_50_cos"],ruthe=False,forecast_year=2022):
    import glob
    #var_list = [
     #   "temp", "press_sl", "humid"]
    #var_list= [ "press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50",     "rain", "wind_10", "wind_50","wind_dir_50_sin", "temp"]
    forecast_year=forecast_year
    dtl = datetime.datetime(forecast_year - 1, 12, 31, 23)
    dtlast = dtl - datetime.timedelta(hours=window_size - 1)
    dtlast = dtl - datetime.timedelta(hours=window_size - 1)
    if ruthe:
       nc_path = '../Data/ruthe_' + str(
            forecast_year)+'_resample_stunden.nc'
       nc_path_last = '../Data/ruthe_' + str(
            forecast_year-1)+'_resample_stunden.nc'
    else:
        nc_path = '../Data/stunden/' + str(
            forecast_year) + '_resample_stunden.nc'  # Replace with the actual path to your NetCDF file
        nc_path_last = '../Data/stunden/' + str(forecast_year - 1) + '_resample_stunden.nc'
    data = xr.open_dataset(nc_path)  # .to_dataframe()#["index">dt]
    datalast = xr.open_dataset(nc_path_last)
    # print(data.to_dataframe().iloc[-1])
    data = xr.concat([datalast, data], dim="index").to_dataframe()
    start_index_forecast = data.index.get_loc(dtlast)
    forecast_data = data[start_index_forecast:]
    #print(forecast_data.head())
    print((len(forecast_data.index.tolist()) - window_size)/forecast_horizont)

    data = {}
    for forecast_var in var_list:
        print(forecast_var)
        lost_window=0
        pred_lis = []
        multivariant_model_path = model_folder#+"/best_model_state_" + forecast_var +'_'+str(window_size)+'_'+str(forecast_horizont)+ ".pt"#../Model/opti/output/lstm_multi/models/best_model_state_" + forecast_var + ".pt"
        lstm_multi_params = params_folder#+'/best_params_lstm_multi_' + forecast_var + '.yaml'#'../Model/opti/output/lstm_multi/best_params_lstm_multi_' + forecast_var + '.yaml'
        for window, last_window in zip(range(window_size, len(forecast_data.index.tolist()), forecast_horizont),range(0, len(forecast_data.index.tolist()) - window_size,forecast_horizont)):
            #print(len(forecast_data[last_window:window]))
            predictions_multi = multilstm_full(multivariant_model_path, forecast_data, start_idx=last_window,
                                               end_idx=window, forecast_var=forecast_var,hyper_params_path=lstm_multi_params,forecast_horizon=forecast_horizont,window_size=window_size,cor_vars=corvars,numvars=len(corvars))
            pred_lis.append(predictions_multi)
            lost_window += 1
            #lost_last_window = last_window
        data[forecast_var]=np.array(pred_lis).flatten()

    df = pd.DataFrame(data)
    if len(df)>len(visual_data):

        df=df[:-(len(df)-len(visual_data))]
    #print(lost_window)
    #print(df)
    output_file = output_file#"forecast_lstm_multi.nc"
    df = df.set_index(pd.to_datetime(visual_data.index.tolist()), inplace=False)
    df.index.name = "Datum"
    #df = xr.Dataset.from_dataframe(df).rename({forecast_var: str(window_size) +'_'+str(forecast_horizont)+'_'+forecast_var})
    df.to_xarray().to_netcdf(output_file)


