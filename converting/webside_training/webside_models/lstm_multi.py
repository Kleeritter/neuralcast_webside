# Import necessary libraries

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import random
import xarray as xr
from scipy import stats


# Set random seeds for reproducibility
pl.seed_everything(42)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Define a custom dataset class for multivariate temperature forecasting
class LSTM_MULTI_Dataset(Dataset):
    def __init__(self, file_path,forecast_horizont=24,window_size=24,forecast_var="temp"):
        # Load data from NetCDF file and preprocess
        self.data = xr.open_dataset(file_path).to_dataframe()
        self.length = len(self.data[forecast_var]) - window_size
        # Set dataset parameters
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

        # Ensure the target has the required length
        if target.shape[0] < self.forecast_horizont:
            target = np.pad(target, ((0, self.forecast_horizont - target.shape[0])), mode='constant')

        # Convert data to torch tensors
        target = np.array(target).reshape((self.forecast_horizont,))
        window_data = torch.from_numpy(np.array(window_data)).float()#[:, np.newaxis]).float()
        target = torch.from_numpy(target).float()
        #print(window_data.shape)
        return window_data, target


# Define a PyTorch Lightning module for multivariate temperature forecasting

class LSTM_MULTI_Model(pl.LightningModule):
    def __init__(self, window_size=24, forecast_horizont=24, num_layers=1, num_lin_layers=2,num_lstm_layers=2,lin_layer_dim=48,hidden_size=40, learning_rate=0.001, weight_decay=0.001, weight_initializer="None", numvars=12):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_lin_layers =num_lin_layers
        self.num_lstm_layers =num_lstm_layers
        self.lin_layer_dim =lin_layer_dim
        self.lstm = torch.nn.LSTM(input_size=numvars, hidden_size=hidden_size, num_layers=num_layers,batch_first=True)
        if num_lstm_layers==2:
            self.lstm2 = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,batch_first=True)
        if num_lin_layers==0:
            self.final_linear = torch.nn.Linear(hidden_size, forecast_horizont)
        else:
            self.first_linear =torch.nn.Linear(hidden_size, lin_layer_dim)
            self.linears = torch.nn.ModuleList([torch.nn.Linear(lin_layer_dim, lin_layer_dim) for _ in range(1,num_lin_layers)])
            self.final_linear = torch.nn.Linear(lin_layer_dim, forecast_horizont)


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
        if self.num_lstm_layers==2:
            lstm_output, _  = self.lstm2(lstm_output)
 
        if self.num_lin_layers==0:
                output = self.final_linear(lstm_output[:,-1, :])
        else:
            linear_output = self.first_linear(lstm_output[:, -1, :])
            for linear_layer_index in range(len(self.linears)):
                linear_output = self.linears[linear_layer_index](linear_output[:, :])
            output = self.final_linear(linear_output[:, :])
        #middle = self.linear(lstm_output[:, -1, :])
        #output = self.linear2(middle[:, :])

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
        self.log('val_loss', loss, prog_bar=False)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)  # weight_decay-Wert anpassen
        return optimizer
