import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
from webside_models.lstm_multi import LSTM_MULTI_Dataset, LSTM_MULTI_Model
import yaml
from  tqdm  import tqdm


file_path = "converting/webside_training/normal_2016-2022.nc"
logs ="/home/alex/Dokumente/lightning_logs"

def train(forecast_var,forecast_horizont,Ruthe=False):
    # Load best hyperparameters from file
    with open('converting/webside_training/webside_params/multi/best_params_lstm_multi_herrenhausen_Temperatur.yaml') as file :#'webside_training/webside_params/multi/best_params_lstm_multi_'+forecast_var+'.yaml') as file:
        best_params = yaml.load(file, Loader=yaml.FullLoader)

    # Create the dataset using TemperatureDataset_multi class

    # Initialize the model with the suggested hyperparameters
    dataset = LSTM_MULTI_Dataset(file_path, forecast_horizont=forecast_horizont, window_size=best_params["window_size"],
                                       forecast_var=forecast_var)
    # Split dataset into training and validation sets
    train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)

    model = LSTM_MULTI_Model(hidden_size=best_params['hidden_size'], learning_rate=best_params['learning_rate'], weight_decay=best_params['weight_decay'],
                                    num_layers=best_params['num_layers'], weight_initializer=best_params['weight_initializer'],forecast_horizont=forecast_horizont,window_size=best_params['window_size'],
                                    num_lin_layers=best_params["lin_layer_num"],lin_layer_dim=best_params["lin_layer_dim"],num_lstm_layers=best_params["lstm_layer_num"],numvars=16)

    # Set up TensorBoardLogger and trainer
    logger = loggers.TensorBoardLogger(save_dir=logs+'/lstm_multi/' + forecast_var, name='lstm_optimierer')
    trainer = pl.Trainer(logger=logger, max_epochs=20, accelerator="auto", devices="auto",
                         deterministic=True, enable_progress_bar=True, callbacks=[EarlyStopping(monitor='val_loss', patience=5, mode='min')])



    # Create data loaders for training and validation
    train_loader = DataLoader(train_data, batch_size=best_params['batchsize'], shuffle=False, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=best_params['batchsize'], shuffle=False, num_workers=8)

    # Fit the model on the training data

    trainer.fit(model, train_loader, val_loader)

    # Save the trained model
    if Ruthe:
        torch.save(model.state_dict(), 'Ruthe/lstm_multi_red/models/best_model_state_'+forecast_var+'_'+str(window_size)+'_'+str(forecast_horizont)+'.pt')
    else:
        torch.save(model.state_dict(), 'converting/2webside_training/saved_models/multi/best_model_state_'+forecast_var+'.pt')
    return



train("herrenhausen_Temperatur",forecast_horizont=24)