def start_index_real(nc_path,gesuchtes_datum,window_size=24,forecast_horizon=24):
    import xarray as xr
    data = xr.open_dataset(nc_path)
    dataf = data.to_dataframe()
    start_index_real = dataf.index.get_loc(gesuchtes_datum)-window_size
    return start_index_real
def end_index_real(nc_path,gesuchtes_datum,window_size=24,forecast_horizon=24):
    import xarray as xr
    data = xr.open_dataset(nc_path)
    dataf = data.to_dataframe()
    end_index_real = dataf.index.get_loc(gesuchtes_datum) +forecast_horizon
    return end_index_real
def start_index_test(nc_path,gesuchtes_datum,window_size=24,forecast_horizon=24):
    import xarray as xr
    data = xr.open_dataset(nc_path)
    dataf = data.to_dataframe()
    start_index_test= dataf.index.get_loc(gesuchtes_datum) -window_size
    return start_index_test
def end_index_test(nc_path,gesuchtes_datum,window_size=24,forecast_horizon=24):
    import xarray as xr
    data = xr.open_dataset(nc_path)
    dataf = data.to_dataframe()
    end_index_test= dataf.index.get_loc(gesuchtes_datum)
    return end_index_test
import numpy as np
from sklearn.metrics import mean_squared_error
import yaml
import math


np.random.seed(42)
def skill_score(actual_values, prediction, reference_values):
    forecast = math.sqrt(mean_squared_error(actual_values, prediction))
    reference = math.sqrt(mean_squared_error(actual_values, reference_values))

    sc=1-(forecast/reference)
    return sc


def lstm_uni(modell,real_valueser,start_index, end_index,forecast_horizon=24,window_size=24, hyper_params_path="../opti/output/lstm_single/best_params_lstm_singletemp_org.yaml",forecast_var="temp"):
    from Model.funcs.funcs_lstm_single import TemperatureModel
    import numpy as np
    import torch
    from sklearn.preprocessing import MinMaxScaler
    checkpoint_path = modell
    checkpoint = torch.load(checkpoint_path)
    optimizer = "Adam"
    dropout = 0  # 0.5
    weight_initializer = "kaiming"
    hyper_params=load_hyperparameters(hyper_params_path)
    model  = TemperatureModel(hidden_size=hyper_params["hidden_size"], learning_rate=hyper_params["learning_rate"], weight_decay=hyper_params["weight_decay"],optimizer=optimizer,num_layers=hyper_params["num_layers"],dropout=dropout,weight_intiliazier=hyper_params["weight_intiliazier"])
    model.load_state_dict(checkpoint)  # ['state_dict'])
    model.eval()
    sliding_window = []
    predicted_values = []
    scaler = MinMaxScaler(feature_range=(0, 1))
    param_path = "../Data/params_for_normal.yaml"  # "../../Data/params_for_normal.yaml"
    params = load_hyperparameters(param_path)
    mins = params["Min_" + forecast_var]
    maxs = params["Max_" + forecast_var]
    #print(real_valueser)
    train_values = [mins, maxs]
    X_train_minmax = scaler.fit_transform(np.array(train_values).reshape(-1, 1))
    real_valueser = scaler.transform([[x] for x in real_valueser]).flatten()

    sliding_window= real_valueser[start_index:end_index]#.values


    sliding_window=np.expand_dims(sliding_window, axis=0)
    sliding_window=np.expand_dims(sliding_window, axis=2)
    input_data = torch.from_numpy(sliding_window).float()
    with torch.no_grad():
        predicted_value = model(input_data)
    predicted_values.append(predicted_value.tolist())
    predictions=predicted_value.squeeze().tolist()

    predicted_values=np.array(predicted_values).flatten()
    denormalized_values = scaler.inverse_transform(predicted_values.reshape(-1,1)).flatten()

    return denormalized_values


def multilstm_full(modell,data,start_idx,end_idx,forecast_horizon,window_size,forecast_var="temp",hyper_params_path="../opti/output/lstm_multi/best_params_lstm_multi.yaml",cor_vars=[ "temp","press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50","rain", "wind_10", "wind_50","wind_dir_50_sin", "wind_dir_50_cos"],numvars=12):
    from Model.funcs.funcs_lstm_multi import TemperatureModel_multi_full
    import numpy as np
    import torch
    from sklearn.preprocessing import MinMaxScaler
    from sklearn import preprocessing
    from scipy import stats
    checkpoint_path = modell
    checkpoint = torch.load(checkpoint_path)
    data=data[cor_vars]
    data=data[start_idx: end_idx]
    for column in data.columns:
        values = data[column].values.reshape(-1, 1)
        if column == "srain":
            data[column] = stats.zscore(values).flatten()
            mean = np.mean(values)
            std = np.std(values)
        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
            param_path = "../Data/params_for_normal.yaml"  # '/home/alex/PycharmProjects/nerualcaster/Data/params_for_normal.yaml'  # "../../Data/params_for_normal.yaml"
            params = load_hyperparameters(param_path)
            mins = params["Min_" + column]
            maxs = params["Max_" + column]
            train_values = [mins, maxs]
            X_train_minmax = scaler.fit_transform(np.array(train_values).reshape(-1, 1))
            scaled_values = scaler.transform(values)
            data[column] = scaled_values.flatten()
            if column == forecast_var:
                scalera = scaler

    predicted_values = []

    best_params = load_hyperparameters(hyper_params_path)

    model = TemperatureModel_multi_full(hidden_size=best_params['hidden_size'], learning_rate=best_params['learning_rate'], weight_decay=best_params['weight_decay'],
                                  num_layers=best_params['num_layers'], weight_initializer=best_params['weight_initializer'],forecast_horizont=forecast_horizon,window_size=window_size,numvars=numvars)
    model.load_state_dict(checkpoint)  # ['state_dict'])
    model.eval()
    sliding_window = data
    sliding_window = np.expand_dims(sliding_window, axis=0)

    input_data = torch.from_numpy(np.array(sliding_window)).float()
    with torch.no_grad():
        predicted_value = model(input_data)
    predicted_values = np.array(predicted_value).flatten()

    if forecast_var=="srain":
        denormalized_values = predicted_values * std + mean
    else:
        denormalized_values = scalera.inverse_transform(predicted_values.reshape(-1, 1)).flatten()

    return denormalized_values

def load_hyperparameters(file_path):
    with open(file_path, 'r') as file:
        hyperparameters = yaml.safe_load(file)
    return hyperparameters

