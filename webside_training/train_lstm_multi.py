import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
from Model.funcs.funcs_lstm_multi import TemperatureDataset_multi, TemperatureModel_multi_full
from Visualistion.createframes import forecast_lstm_multi
import yaml
from  tqdm  import tqdm

full='../Data/zusammengefasste_datei_2016-2021.nc'
file_path =full
logs ="/home/alex/Dokumente/lightning_logs"

def train(forecast_var,forecast_horizont,window_size,transfer_learning=False,corvars=[ "temp","press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50",     "rain", "wind_10", "wind_50","wind_dir_50_sin", "wind_dir_50_cos","taupunkt","Taupunkt3h","press3h","rainsum3h","temp3h","gradwind","rain_event"],Ruthe=False):
    # Load best hyperparameters from file
    with open('opti/output/lstm_multi/best_params_lstm_multi_'+forecast_var+'.yaml') as file:
        best_params = yaml.load(file, Loader=yaml.FullLoader)

    # Create the dataset using TemperatureDataset_multi class

    dataset = TemperatureDataset_multi(file_path, forecast_horizont=forecast_horizont, window_size=window_size,
                                       forecast_var=forecast_var,cor_vars=corvars)
    # Split dataset into training and validation sets
    train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)

    if transfer_learning:
        model = TemperatureModel_multi_full(hidden_size=best_params['hidden_size'],
                                            learning_rate=best_params['learning_rate'],
                                            weight_decay=best_params['weight_decay'],
                                            num_layers=best_params['num_layers'],
                                            weight_initializer=best_params['weight_initializer'],
                                            forecast_horizont=forecast_horizont, window_size=window_size)
        checkpoint = torch.load('timetest/lstm_multi/models/best_model_state_'+forecast_var+'_'+str(5*24)+'_'+str(forecast_horizont)+'.pt')
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = TemperatureModel_multi_full(hidden_size=best_params['hidden_size'], learning_rate=best_params['learning_rate'], weight_decay=best_params['weight_decay'],
                                    num_layers=best_params['num_layers'], weight_initializer=best_params['weight_initializer'],forecast_horizont=forecast_horizont, window_size=window_size,numvars=len(corvars))

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
        torch.save(model.state_dict(), 'timetest/lstm_multi/models/best_model_state_'+forecast_var+'_'+str(window_size)+'_'+str(forecast_horizont)+'.pt')
    return
