import torch
import pytorch_lightning as pl
from Model.funcs.funcs_lstm_multi import TemperatureModel_multi_full
best_params = load_hyperparameters(hyper_params_path)

forecast_horizon=24
window_size=24
numvars =12
checkpoint_path = "../Model/"
checkpoint = torch.load(checkpoint_path)
# Passe die Architektur deines LSTM-Modells entsprechend an
model = TemperatureModel_multi_full(hidden_size=best_params['hidden_size'], learning_rate=best_params['learning_rate'],
                                    weight_decay=best_params['weight_decay'],
                                    num_layers=best_params['num_layers'],
                                    weight_initializer=best_params['weight_initializer'],
                                    forecast_horizont=forecast_horizon, window_size=window_size, numvars=numvars)
model.load_state_dict(checkpoint)  # ['state_dict'])
model.eval()