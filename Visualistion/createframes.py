import matplotlib.pyplot as plt
from Model.funcs.visualer_funcs import lstm_uni, skill_score,multilstm_full,tft,ttft
import pandas as pd
import xarray as xr
import datetime
import seaborn as sns
import numpy as np
from Model.funcs.trad import p_ro,sarima,arima
from tqdm import tqdm
from multiprocessing import Pool
# Passe die folgenden Variablen entsprechend an



forecast_var="temp" #Which variable to forecast
window_size=24*7*4 #How big is the window for training
forecast_horizon=24 #How long to cast in the future
forecast_year=2022 #Which year to forecast
dt = datetime.datetime(forecast_year,1,1,0,0) #+ datetime.timedelta(hours=window_size)
dtl=datetime.datetime(forecast_year -1 ,12,31,23)
#print(dt +datetime.timedelta(hours=8760))
dtlast= dtl - datetime.timedelta(hours=window_size-1)
nc_path = '../Data/stunden/'+str(forecast_year)+'_resample_stunden.nc' # Replace with the actual path to your NetCDF file
nc_path_last = '../Data/stunden/'+str(forecast_year-1)+'_resample_stunden.nc'
data = xr.open_dataset(nc_path)#.to_dataframe()#["index">dt]
datalast= xr.open_dataset(nc_path_last)
#print(data.to_dataframe().iloc[-1])
data=xr.concat([datalast,data],dim="index").to_dataframe()
start_index_forecast = data.index.get_loc(dtlast)
start_index_visual = data.index.get_loc(dt)
forecast_data=data[start_index_forecast:]
#print(forecast_data)
visual_data=data[start_index_visual:]
#print(data[:start_index_visual+1])
#print(data)
datus= xr.open_dataset(nc_path)
#print(forecast_data[0:forecast_data.index.get_loc(dt)+1])
univariant_model_path = '../Model/opti/output/lstm_single/models/best_model_state_temp.pt'#'../Model/output/lstm_uni/'+forecast_var+'optimierter.pth' # Replace with the actual path to your model
multivariant_model_path = '../Model/output/lstm_multi/'+forecast_var+'_unoptimiert.pth' # Replace with the actual path to your model
tft_model_path='../Model/output/tft/'+forecast_var+'._unoptimiert.pth' # Replace with the actual path to your model




learning_rate =0.00005
weight_decay = 0.0001
hidden_size = 32
optimizer= "Adam"
num_layers=1
dropout=0
#print("koksnot")
start_index_test = forecast_data.index.get_loc(dt)

#references=np.load("sarima/reference_temp_.npy").flatten()

#print(forecast_data.iloc[-1])


predicted_temp_tft=[]
var_list = ["wind_dir_50_sin", "wind_dir_50_cos", 'temp', "press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10",
            "gust_50", "rain", "wind_10", "wind_50"]
numbers = np.arange(0, 12)
empty_lists = {var: [] for var in var_list}

def fullsarima(number):
    import datetime
    import xarray as xr
    import numpy as np

    var=var_list[number]
    print(var)
    referencor=[]
    for window, last_window in tqdm(zip(range(window_size, len(forecast_data.index.tolist()), forecast_horizon),
                                   range(0, len(forecast_data.index.tolist()) - window_size,
                                         forecast_horizon))):
        refenrence = sarima.sarima(forecast_data[forecast_var][last_window:window])
        referencor.append(refenrence)
        np.save(file="sarima/reference" + var, arr=np.array(referencor))
    return




def litesarima():
    var_list = [
        "diffuscmp11","globalrcmp11"]  # , "Geneigt CM-11", 'temp', "press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50","rain", "wind_10", "wind_50"]

    for var in var_list:
        print(var)
        referencor=[]
        for window, last_window in tqdm(zip(range(window_size, len(forecast_data.index.tolist()), forecast_horizon),
                                       range(0, len(forecast_data.index.tolist()) - window_size,
                                             forecast_horizon))):
            #print(forecast_data[var][last_window:window])
            refenrence = sarima.sarima(forecast_data[var][last_window:window])
            referencor.append(refenrence)
        np.save(file="sarima/reference"+var,arr=np.array(referencor))
    return

def lite_arima():
    var_list = [
        "rain","gust_10","press_sl","humid","gust_50","wind_10","wind_50","wind_dir_50_sin","wind_dir_50_cos","diffuscmp11"]  # , "Geneigt CM-11", 'temp', "press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50","rain", "wind_10", "wind_50"]

    for var in var_list:
        print(var)
        referencor = []
        for window, last_window in tqdm(zip(range(window_size, len(forecast_data.index.tolist()), forecast_horizon),
                                            range(0, len(forecast_data.index.tolist()) - window_size,
                                                  forecast_horizon))):
            # print(forecast_data[var][last_window:window])
            refenrence = arima.arima(forecast_data[var][last_window:window])
            referencor.append(refenrence)
        np.save(file="sarima/reference" + var, arr=np.array(referencor))
    return

def sarima_netcdf():
    var_list = [
        "wind_dir_50_sin","temp", "wind_dir_50_cos","rain", "press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50", "wind_10", "wind_50"]  # , "Geneigt CM-11", 'temp', "press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50","rain", "wind_10", "wind_50"]
    data = {}
    for var in var_list:
        print(var)
        references_var = np.load("sarima/reference"+var+".npy").flatten()
        data[var] = references_var
        print(data[var])
    df = pd.DataFrame(data)

    output_file = "forecast_sarima.nc"
    df = df.set_index(pd.to_datetime(visual_data.index.tolist()), inplace=False)
    df.index.name = "Datum"
    df = xr.Dataset.from_dataframe(df)
    df.to_netcdf(output_file)

def forecast_lstm_uni():

    var_list = [
        "temp", "press_sl", "humid"]
    data = {}
    for forecast_var in var_list:
        print(forecast_var)
        pred_lis = []
        univariant_model_path= "../Model/opti/output/lstm_single/models/best_model_state_"+forecast_var+".pt"
        lstm_uni_params = '../Model/opti/output/lstm_single/best_params_lstm_single_'+forecast_var+'.yaml'
        for window, last_window in zip(range(window_size, len(forecast_data.index.tolist()), forecast_horizon),
                                       range(0, len(forecast_data.index.tolist()) - window_size,
                                             forecast_horizon)):

            predictions = lstm_uni(univariant_model_path, forecast_data[forecast_var], start_index=last_window,
                                   end_index=window,hyper_params_path=lstm_uni_params,forecast_var=forecast_var)  # .insert(0, Messfrühling[0:24]),
            pred_lis.append(predictions)
        data[forecast_var]=np.array(pred_lis).flatten()

    df = pd.DataFrame(data)
    print(df)
    output_file = "forecast_lstm_uni.nc"
    df = df.set_index(pd.to_datetime(visual_data.index.tolist()), inplace=False)
    df.index.name = "Datum"
    df=xr.Dataset.from_dataframe(df)
    df.to_netcdf(output_file)




def forecast_lstm_multi(model_folder,params_folder,var_list,output_file,forecast_horizont,window_size,corvars=[ "temp","press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50","rain", "wind_10", "wind_50","wind_dir_50_sin", "wind_dir_50_cos"]):
    import glob
    #var_list = [
     #   "temp", "press_sl", "humid"]
    #var_list= [ "press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50",     "rain", "wind_10", "wind_50","wind_dir_50_sin", "temp"]
    forecast_year=2022
    dtl = datetime.datetime(forecast_year - 1, 12, 31, 23)
    dtlast = dtl - datetime.timedelta(hours=window_size - 1)
    dtlast = dtl - datetime.timedelta(hours=window_size - 1)
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

def forecast_tft():
    predicted_temp_multi = []
    for window, last_window in zip(range(window_size, len(forecast_data.index.tolist()), forecast_horizon),
                                   range(0, len(forecast_data.index.tolist()) - window_size,
                                         forecast_horizon)):
        predictions_multi = tft(tft_model_path, forecast_data, start_idx=last_window,
                                           end_idx=window, forecast_var=forecast_var)
        predicted_temp_multi.append(predictions_multi)
    data = pd.DataFrame({
        'temp': np.array(predicted_temp_multi).flatten(),

    })
    output_file = "forecast_tft.nc"
    df = data.set_index(pd.to_datetime(visual_data.index.tolist()), inplace=False)
    df.index.name = "Datum"
    df=xr.Dataset.from_dataframe(df)
    df.to_netcdf(output_file)

def forecast_ttft():
    var_list = ["temp"]  # , "Geneigt CM-11", 'temp', "press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50","rain", "wind_10", "wind_50"]
    data = {}
    for forecast_var in var_list:
        print(forecast_var)
        pred_lis = []
        #model_path = "../Model/output/ttft/" + forecast_var + ".ckpt"
        model_path="../Model/lightning_logs/lightning_logs/version_11/checkpoints/epoch=49-step=2500.ckpt"
        #lstm_uni_params = '../Model/opti/output/lstm_single/best_params_lstm_single_' + forecast_var + '.yaml'
        for window, last_window in zip(range(window_size, len(forecast_data.index.tolist()), forecast_horizon),
                                       range(0, len(forecast_data.index.tolist()) - window_size,
                                             forecast_horizon)):
            predictions = ttft(model_path, forecast_data, start_index=last_window,
                                   end_index=window,
                                   forecast_var=forecast_var)  # .insert(0, Messfrühling[0:24]),
            pred_lis.append(predictions)
        data[forecast_var] = np.array(pred_lis).flatten()
    df = pd.DataFrame(data)
    print(df)
    output_file = "forecast_ttft.nc"
    df = df.set_index(pd.to_datetime(visual_data.index.tolist()), inplace=False)
    df.index.name = "Datum"
    df = xr.Dataset.from_dataframe(df)
    df.to_netcdf(output_file)



#with Pool() as pool:
 #   pool.map(fullsarima, numbers)

#df = pd.DataFrame(empty_lists)

# Setze den Index auf "Datum"
#df = df.set_index(pd.to_datetime(visual_data.index.tolist()), inplace=False)
#output_file = "forecast_sarima.nc"
#df = xr.Dataset.from_dataframe(df)
#df.to_netcdf(output_file)
#forecast_lstm_uni()
#forecast_ttft()
#forecast_lstm_multi()
#litesarima()
#lite_arima()
#sarima_netcdf()
#forecast_tft()
#var= "wind_dir_50"
#references_var = np.load("sarima/reference_"+var+".npy").flatten()
#print(references_var)