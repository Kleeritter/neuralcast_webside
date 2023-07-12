import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
from funcs.funcs_lstm_multi import TemperatureDataset_multi, TemperatureModel_multi_full
from Visualistion.createframes import forecast_lstm_multi
import yaml

full='../Data/zusammengefasste_datei_2016-2019.nc'
file_path =full # Replace with the actual path to your NetCDF file
logs ="/home/alex/Dokumente/lightning_logs"

def train(forecast_var,forecast_horizont,window_size):
    # Lade die Hyperparameter
    with open('opti/output/lstm_multi/best_params_lstm_multi_'+forecast_var+'.yaml') as file:
        best_params = yaml.load(file, Loader=yaml.FullLoader)

    dataset = TemperatureDataset_multi(file_path, forecast_horizont=forecast_horizont, window_size=window_size,
                                       forecast_var=forecast_var)
    train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)
    train_loader = DataLoader(train_data, batch_size=best_params["batchsize"], shuffle=False, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=best_params["batchsize"], shuffle=False, num_workers=8)
    # Trainiere das Modell mit den besten Parametern
    model = TemperatureModel_multi_full(hidden_size=best_params['hidden_size'], learning_rate=best_params['learning_rate'], weight_decay=best_params['weight_decay'],
                                  num_layers=best_params['num_layers'], weight_initializer=best_params['weight_initializer'],forecast_horizont=forecast_horizont, window_size=window_size)
    logger = loggers.TensorBoardLogger(save_dir=logs+'/lstm_multi/' + forecast_var, name='lstm_optimierer')
    trainer = pl.Trainer(logger=logger, max_epochs=1, accelerator="auto", devices="auto",
                         deterministic=True, enable_progress_bar=True)



    # Create data loaders for training and validation
    train_loader = DataLoader(train_data, batch_size=best_params['batchsize'], shuffle=False, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=best_params['batchsize'], shuffle=False, num_workers=8)
    trainer.fit(model, train_loader, val_loader)

    # Speichere den Modellzustand des besten Modells
    torch.save(model.state_dict(), 'timetest/lstm_multi/models/best_model_state_'+forecast_var+'_'+str(window_size)+'_'+str(forecast_horizont)+'.pt')
    return

if __name__ == '__main__':
    forecast_vars=['temp']
    for forecast_var in forecast_vars:
        for window_size in [4*7*24]:
            for forecast_horizont in [12]:
                train('temp',forecast_horizont,window_size)
                model_path='timetest/lstm_multi/models'
                params_folder='opti/output/lstm_multi/best_params_lstm_multi_'+"temp"+'.yaml'
                varlist = ['temp']
                output_file='timetest/lstm_multi/output/'+forecast_var+'/timetest_lstm_multi'+forecast_var+'_'+str(window_size)+'_'+str(forecast_horizont)+'.nc'
                forecast_lstm_multi(var_list=varlist, model_folder=model_path, params_folder=params_folder,forecast_horizont=forecast_horizont,window_size=window_size,output_file=output_file)