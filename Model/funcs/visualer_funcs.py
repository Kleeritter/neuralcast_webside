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

# Setze den Random Seed für numpy
np.random.seed(42)
def skill_score(actual_values, prediction, reference_values):
    forecast = math.sqrt(mean_squared_error(actual_values, prediction))#np.sqrt(mean_squared_error(actual_values, prediction))
    reference = math.sqrt(mean_squared_error(actual_values, reference_values))#np.sqrt(mean_squared_error(actual_values, reference_values))
    perfect_forecast =1# np.sqrt(mean_squared_error(actual_values, actual_values))

    #sc= (forecast- reference) / (perfect_forecast - reference)
    sc=1-(forecast/reference)
    return sc


def lstm_uni(modell,real_valueser,start_index, end_index,forecast_horizon=24,window_size=24, hyper_params_path="../opti/output/lstm_single/best_params_lstm_singletemp_org.yaml",forecast_var="temp"):
    from Model.funcs.funcs_lstm_single import TemperatureModel
    import numpy as np
    import torch
    from sklearn.preprocessing import MinMaxScaler
    checkpoint_path = modell
    checkpoint = torch.load(checkpoint_path)
    #window_size = 24 * 7 * 4  # 168
    #learning_rate = 0.00028321349862445627  # 0.00005#
    #weight_decay = 6.814701853104705e-05  # 0.0001#
    #hidden_size = 64  # 32
    optimizer = "Adam"
    #num_layers = 2  # 1
    dropout = 0  # 0.5
    weight_initializer = "kaiming"
    hyper_params=load_hyperparameters(hyper_params_path)
    model  = TemperatureModel(hidden_size=hyper_params["hidden_size"], learning_rate=hyper_params["learning_rate"], weight_decay=hyper_params["weight_decay"],optimizer=optimizer,num_layers=hyper_params["num_layers"],dropout=dropout,weight_intiliazier=hyper_params["weight_intiliazier"])
    model.load_state_dict(checkpoint)  # ['state_dict'])
    model.eval()
    sliding_window = []  # Liste für das Sliding Window

    # Führe die Vorhersage für die ersten 24 Stunden durch
    predicted_values = []
    scaler = MinMaxScaler(feature_range=(0, 1))
    param_path = '/home/alex/PycharmProjects/nerualcast/Data/params_for_normal.yaml'  # "../../Data/params_for_normal.yaml"
    params = load_hyperparameters(param_path)
    mins = params["Min_" + forecast_var]
    maxs = params["Max_" + forecast_var]
    #print(real_valueser)
    train_values = [mins, maxs]
    X_train_minmax = scaler.fit_transform(np.array(train_values).reshape(-1, 1))
    real_valueser = scaler.transform([[x] for x in real_valueser]).flatten()

    sliding_window= real_valueser[start_index:end_index]#.values
    #print(len(sliding_window))
    #original_mean = np.mean(sliding_window)  # original_data sind die nicht normalisierten Daten
    #original_std = np.std(sliding_window)
    #sliding_window = (sliding_window - original_mean) / original_std
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #sliding_window = scaler.fit_transform([[x] for x in sliding_window]).flatten()
    # Berechnung des Durchschnitts und der Standardabweichung des Originaldatensatzes

    sliding_window=np.expand_dims(sliding_window, axis=0)
    sliding_window=np.expand_dims(sliding_window, axis=2)
    input_data = torch.from_numpy(sliding_window).float()
    with torch.no_grad():
        predicted_value = model(input_data)
    predicted_values.append(predicted_value.tolist())
    predictions=predicted_value.squeeze().tolist()
    #denormalized_values = [(predicted_value * original_std )+ original_mean for predicted_value in predictions]
    #print(np.array(predicted_values).flatten())
    predicted_values=np.array(predicted_values).flatten()
    denormalized_values = scaler.inverse_transform(predicted_values.reshape(-1,1)).flatten()
    #denormalized_values =predicted_values
    return denormalized_values


def multilstm_full(modell,data,start_idx,end_idx,forecast_horizon,window_size,forecast_var="temp",hyper_params_path="../opti/output/lstm_multi/best_params_lstm_multi.yaml",cor_vars=[ "temp","press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50","rain", "wind_10", "wind_50","wind_dir_50_sin", "wind_dir_50_cos"],numvars=12):
    from Model.funcs.funcs_lstm_multi import TemperatureModel_multi_full
    import numpy as np
    import torch
    from sklearn.preprocessing import MinMaxScaler
    from sklearn import preprocessing
    checkpoint_path = modell
    checkpoint = torch.load(checkpoint_path)
    data=data[cor_vars]
    data=data[start_idx: end_idx]
    #print(data)
    for column in data.columns:
        values = data[column].values.reshape(-1, 1)
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
    #print(data)
    predicted_values = []
    #print(data)
    best_params = load_hyperparameters(hyper_params_path)
    # Passe die Architektur deines LSTM-Modells entsprechend an
    model = TemperatureModel_multi_full(hidden_size=best_params['hidden_size'], learning_rate=best_params['learning_rate'], weight_decay=best_params['weight_decay'],
                                  num_layers=best_params['num_layers'], weight_initializer=best_params['weight_initializer'],forecast_horizont=forecast_horizon,window_size=window_size,numvars=numvars)
    model.load_state_dict(checkpoint)  # ['state_dict'])
    model.eval()
    sliding_window = data  # Liste für das Sliding Window
    sliding_window = np.expand_dims(sliding_window, axis=0)
    #print(forecast_horizon)
    #sliding_window = np.expand_dims(sliding_window, axis=2)
    input_data = torch.from_numpy(np.array(sliding_window)).float()
    with torch.no_grad():
        predicted_value = model(input_data)
    predicted_values = np.array(predicted_value).flatten()
    #print(predicted_values)
    denormalized_values = scalera.inverse_transform(predicted_values.reshape(-1, 1)).flatten()
    #print(len(denormalized_values))
    return denormalized_values

def tft(modell,data,start_idx,end_idx,forecast_horizon=24,window_size=24, forecast_var="temp"):
    from Model.funcs.funcs_tft import TFT_Modell
    import numpy as np
    import torch
    from sklearn.preprocessing import MinMaxScaler
    checkpoint_path = modell
    checkpoint = torch.load(checkpoint_path)
    data = data[["wind_dir_50", "Geneigt CM-11", 'temp', "press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10",
                 "gust_50", "rain", "wind_10", "wind_50"]]
    # print(data.columns)
    for column in data.columns:
        if column == "wind_dir_50":
            # Extrahiere die Windrichtungen
            wind_directions_deg = data[column].values

            # Konvertiere die Windrichtungen in Bogenmaß
            wind_directions_rad = np.deg2rad(wind_directions_deg)

            # Berechne die Sinus- und Kosinus-Werte der Windrichtungen
            sin_directions = np.sin(wind_directions_rad)
            cos_directions = np.cos(wind_directions_rad)

            # Kombiniere Sinus- und Kosinus-Werte zu einer einzigen Spalte
            combined_directions = np.arctan2(sin_directions, cos_directions)

            # Skaliere die kombinierten Werte auf den Bereich von 0 bis 1
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_directions = scaler.fit_transform(combined_directions.reshape(-1, 1)).flatten()

            data.loc[:, column] = scaled_directions

            if forecast_var == column:
                forecast_var_scaler = scaler
        else:
            # Erstelle einen neuen Min-Max-Scaler für jede Spalte
            scaler = MinMaxScaler()

            # Extrahiere die Werte der aktuellen Spalte und forme sie in das richtige Format um
            values = data[column].values.reshape(-1, 1)

            # Skaliere die Werte der Spalte
            scaled_values = scaler.fit_transform(values)

            # Aktualisiere die Daten mit den skalierten Werten
            data.loc[:, column] = scaled_values.flatten()

            if forecast_var == column:
                forecast_var_scaler = scaler


    window_size = 4 * 7 * 24
    forecast_horizont = 24
    input_dim = 12  # Anzahl der meteorologischen Parameter
    output_dim = 1  # Vorhersage der Temperatur

    num_layers = 2
    num_heads = 4
    batch_size = 24
    hidden_dim = 12  # 3*num_heads
    num_encoder_layers = 3
    num_decoder_layers = 3
    d_model = 12
    dropout = 0.1
    patience = 10
    max_epochs = 200

    # Passe die Architektur deines LSTM-Modells entsprechend an
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TFT_Modell(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                       num_heads=num_heads, forecast_horizont=forecast_horizont, window_size=window_size,dropout=dropout).to(device)
    model.load_state_dict(checkpoint)  # ['state_dict'])
    model.eval()
    sliding_window = []  # Liste für das Sliding Window

    # Führe die Vorhersage für die ersten 24 Stunden durch
    predicted_values = []

    sliding_window = data[start_idx: end_idx]
    sliding_window = np.expand_dims(sliding_window, axis=0)
    # print(sliding_window.shape)
    input_data = torch.from_numpy(np.array(sliding_window)).float().to(device)
    # print(input_data.shape)
    with torch.no_grad():
        predicted_value = model(input_data)
    predicted_values.append(predicted_value.tolist())
    predictions = predicted_value.squeeze().tolist()
    # print(predicted_value.shape)
    predicted_values = np.array(predicted_values).flatten()
    # print(predicted_values)
    denormalized_values = forecast_var_scaler.inverse_transform(predicted_values.reshape(-1, 1)).flatten()
    # denormalized_values = [(predicted_value * stds[0] )+ means[0] for predicted_value in predictions]
    # denormalized_values=np.arange(0,len(denormalized_values))
    # print(denormalized_values)
    return denormalized_values



def ttft(modell,data,start_index, end_index,forecast_horizon=24,window_size=24, hyper_params_path="../opti/output/lstm_single/best_params_lstm_singletemp_org.yaml",forecast_var="temp"):
    import numpy as np
    import torch
    from sklearn.preprocessing import MinMaxScaler
    import lightning.pytorch as pl
    from pytorch_forecasting import Baseline, TemporalFusionTransformer,TimeSeriesDataSet
    import warnings

    warnings.filterwarnings("ignore")  # avoid printing out absolute paths

    #hyper_params = load_hyperparameters(hyper_params_path)
    pl.seed_everything(42)
    data=data[start_index:end_index]
    data["timestep"] = np.arange(0,
                                 len(data.index.tolist()))  # pd.to_numeric(data.index)- pd.to_numeric(data.index).min()
    data["const"] = np.ones(len(data))
    data = data.reset_index()
    data = data.drop(columns="index")
    feature_cols = ['humid', 'temp', 'press_sl', 'wind_10', 'wind_50', 'gust_10', 'gust_50', 'rain', 'diffuscmp11',
                    'globalrcmp11', 'wind_dir_50_sin', 'wind_dir_50_cos']  # Liste der meteorologischen Parameter
    for column in feature_cols:
        values = data[column].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        param_path = "../Data/params_for_normal.yaml"  # '/home/alex/PycharmProjects/nerualcaster/Data/params_for_normal.yaml'  # "../../Data/params_for_normal.yaml"
        params = load_hyperparameters(param_path)
        mins = params["Min_" + column]
        maxs = params["Max_" + column]
        train_values = [mins, maxs]
        X_train_minmax = scaler.fit_transform(np.array(train_values).reshape(-1, 1))
        scaled_values = scaler.transform(values)
        data[column] = scaled_values.flatten()
        if column==forecast_var:
            scalera=scaler
    data = data[
        ['humid', 'temp', 'press_sl', 'wind_10', 'wind_50', 'gust_10', 'gust_50', 'rain', 'diffuscmp11', 'globalrcmp11',
         'wind_dir_50_sin', 'wind_dir_50_cos', "timestep", "const"]]
    pred = TimeSeriesDataSet(
        data=data,
        target=forecast_var,
        group_ids=["const"],  # Annahme: Spalte mit Zeitstempeln heißt "Timestamp"
        time_idx="timestep",
        min_prediction_length=1,
        time_varying_unknown_reals=feature_cols,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        predict_mode=True,
        max_prediction_length=24,
        #max_encoder_length=672,
    )

    checkpoint = torch.load(modell)
    #print(checkpoint.keys())
    best_tft = TemporalFusionTransformer.load_from_checkpoint(modell)
    predictions = best_tft.predict(pred,  trainer_kwargs=dict(accelerator="cpu"))
    #print(predictions)
    denormalized_values = scalera.inverse_transform(predictions)
    # denormalized_values =predicted_values
    return denormalized_values


def load_hyperparameters(file_path):
    with open(file_path, 'r') as file:
        hyperparameters = yaml.safe_load(file)
    return hyperparameters

def conv(modell,data,start_idx,end_idx):
    from Model.funcs import TemperatureModel_conv
    import numpy as np
    import torch
    from sklearn import preprocessing
    checkpoint_path = modell
    checkpoint = torch.load(checkpoint_path)
    #print(checkpoint.keys())

    # Passe die Architektur deines LSTM-Modells entsprechend an
    model = TemperatureModel_conv()  # Ersetze "YourLSTMModel" durch den tatsächlichen Namen deines Modells
    model.load_state_dict(checkpoint)  # ['state_dict'])
    model.eval()
    sliding_window = []  # Liste für das Sliding Window
    window_size = 6
    num_channels = 17
    # Führe die Vorhersage für die ersten 24 Stunden durch
    inputs=[]
    predicted_values = []
    for i in range(0,24):
        start_id =start_idx+i
        end_id= start_id+6
        window_data = data.isel(index=slice(start_id, end_id)).to_array().values
        window_data_normalized = np.zeros((window_size, num_channels))
        window_data_normalized[:, 0] = (window_data[0] - np.mean(window_data[0])) / np.std(
            window_data[0])  # air temperature
        window_data_normalized[:, 2] = (window_data[1] - np.mean(window_data[1])) / np.std(
            window_data[1])  # air pressure
        window_data_normalized[:, 1] = (window_data[2] - np.mean(window_data[2])) / np.std(
            window_data[2])  # humid
        window_data_normalized[:, 3] = (window_data[3] - np.mean(window_data[3])) / np.std(
            window_data[3])  # Globalstrahlung
        if np.std(window_data[4]) != 0:
            window_data_normalized[:, 4] = (window_data[4] - np.mean(window_data[4])) / np.std(
                window_data[4])  # gust_10
        else:
            window_data_normalized[:, 4] = np.zeros_like(window_data_normalized[:, 4])

        window_data_normalized[:, 5] = (window_data[5] - np.mean(window_data[5])) / np.std(
            window_data[5])  # gust_50
        if np.std(window_data[6]) != 0:
            window_data_normalized[:, -1] = (window_data[6] - np.mean(window_data[6])) / np.std(
                window_data[6])  # hourly precipitation
        else:
            window_data_normalized[:, -1] = np.zeros_like(window_data_normalized[:, 6])
        if np.std(window_data[7]) != 0:
            window_data_normalized[:, 6] = (window_data[7] - np.mean(window_data[7])) / np.std(
                window_data[7])  # wind_10
        else:
            window_data_normalized[:, 6] = np.zeros_like(window_data_normalized[:, 7])
        window_data_normalized[:, 7] = (window_data[8] - np.mean(window_data[8])) / np.std(
            window_data[8])  # wind_50
        # print(window_data_normalized.shape)
        # One-hot encode wind direction
        wind_direction = window_data[9]
        normalized_directions = wind_direction % 360
        numeric_directions = (normalized_directions / 45).astype(int) % 8
        encoder = preprocessing.OneHotEncoder(categories=[np.arange(8)], sparse_output=False)
        enc = encoder.fit_transform(numeric_directions.reshape(-1, 1))

        window_data_normalized[:, 8:-1] = enc
        #print(window_data_normalized)
        inputs.append(window_data_normalized)
    #print(np.transpose(inputs)[0][0])
    window_data = data.isel(index=slice(start_idx, end_idx)).to_array().values
    stbw=np.std(window_data[0])#np.transpose(inputs)[0][0])
    mean=np.mean(window_data[0])#np.transpose(inputs)[0][0])
    #print(stbw)
    #print(mean)
    #window_data = torch.from_numpy(window_data_normalized).float()

    #sliding_window=np.expand_dims(sliding_window, axis=2)
    #input_data = torch.Tensor(sliding_window)
    tensor=torch.empty((24,6,17))
    for i in range(0,24):
        tensor[i]=torch.tensor(inputs[i])
    input_data = tensor#torch.from_numpy(tensor).float()
    with torch.no_grad():
        predicted_value = model(input_data)
    predicted_values.append(predicted_value.tolist())
    predictions=predicted_value.squeeze().tolist()[-1]
    #print(predictions)
    denormalized_values = [(predicted_value *stbw )+ mean for predicted_value in predictions]
    #denormalized_values=np.arange(0,len(denormalized_values))
    #print(denormalized_values)
    return denormalized_values