from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import random
import xarray as xr
from Model.funcs.visualer_funcs import load_hyperparameters
from scipy import stats
pl.seed_everything(42)

# Setze den Random Seed für torch
torch.manual_seed(42)

# Setze den Random Seed für random
random.seed(42)

# Setze den Random Seed für numpy
np.random.seed(42)


class TemperatureDataset_multi(Dataset):
    def __init__(self, file_path,forecast_horizont=24,window_size=24,forecast_var="temp",cor_vars=["wind_dir_50_sin","wind_dir_50_cos",'temp',"press_sl","humid","diffuscmp11","globalrcmp11","gust_10","gust_50", "rain", "wind_10", "wind_50"]):
        self.data = xr.open_dataset(file_path)[cor_vars].to_dataframe()#.valuesmissing_values_mask = dataset['temp'].isnull()
        self.length = len(self.data[forecast_var]) - window_size
       # scaler = MinMaxScaler(feature_range=(0, 1))
       # self.data=scaler.fit_transform([[x] for x in self.data]).flatten()
        for column in self.data.columns:
            values = self.data[column].values.reshape(-1, 1)
            if column == "srain":
                self.data[column] = stats.zscore(values).flatten()

            else:
                scaler = MinMaxScaler(feature_range=(0, 1))
                param_path ='/home/alex/PycharmProjects/neuralcaster/Data/params_for_normal.yaml'  # "../../Data/params_for_normal.yaml"
                params = load_hyperparameters(param_path)
                mins = params["Min_" + column]
                maxs = params["Max_" + column]
                train_values = [mins, maxs]
                X_train_minmax = scaler.fit_transform(np.array(train_values).reshape(-1, 1))
                #self.data = scaler.transform([[x] for x in self.data]).flatten()
                scaled_values = scaler.transform(values)
                self.data[column] = scaled_values.flatten()

        print(self.data["rain"])

        #print(self.data)
        self.forecast_horizont = forecast_horizont
        self.window_size = window_size
        self.forecast_var = forecast_var

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.window_size
        window_data = self.data.iloc[start_idx:end_idx]#isel(index=slice(start_idx, end_idx)).to_array()#.values#self.data[start_idx:end_idx].values
        target = self.data[self.forecast_var][end_idx:end_idx+self.forecast_horizont]#.values

        # Check if target has exactly 24 hours, otherwise adjust it
        if target.shape[0] < self.forecast_horizont:
            target = np.pad(target, ((0, self.forecast_horizont - target.shape[0])), mode='constant')
        # Convert to torch tensors
        #window_data = window_data_normalized.transpose(1, 0)
        #print(window_data)
        target = np.array(target).reshape((self.forecast_horizont,))
        window_data = torch.from_numpy(np.array(window_data)).float()#[:, np.newaxis]).float()
        target = torch.from_numpy(target).float()
        #print(window_data.shape)
        return window_data, target


class TemperatureModel_multi_light(pl.LightningModule):
    pl.seed_everything(42)

    # Setze den Random Seed für torch
    torch.manual_seed(42)

    # Setze den Random Seed für random
    random.seed(42)

    # Setze den Random Seed für numpy
    np.random.seed(42)
    def __init__(self, window_size=24, forecast_horizont=24):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=3, hidden_size=32, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(32, forecast_horizont)

    def forward(self, x):
        #print(x.shape)
        lstm_output, _ = self.lstm(x)
        output = self.linear(lstm_output[:, -1, :])
        #print(output.shape)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.001)  # weight_decay-Wert anpassen
        return optimizer

class TemperatureModel_multi_full(pl.LightningModule):
    def __init__(self, window_size=24, forecast_horizont=24, num_layers=1, hidden_size=40, learning_rate=0.001, weight_decay=0.001, weight_initializer="None", numvars=12):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lstm = torch.nn.LSTM(input_size=numvars, hidden_size=hidden_size, num_layers=num_layers,batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, forecast_horizont)

        self.weight_initializer = weight_initializer
        self.apply(self.initialize_weights)

    def initialize_weights(self, module):
        match self.weight_initializer:
            case "None":
                pass
            case "kaiming":
                for name, param in self.lstm.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.kaiming_uniform_(param, nonlinearity='relu')
            case "normal":
                for name, param in self.lstm.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.normal_(param)
            case "xavier":
                for name, param in self.lstm.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.xavier_uniform_(param)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        output = self.linear(lstm_output[:, -1, :])
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        #loss = torch.nn.MSELoss()(y_hat, y)
        loss = torch.nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss, prog_bar=False)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)  # weight_decay-Wert anpassen
        return optimizer
