import optuna_dashboard,optuna
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from Model.funcs.funcs_lstm_single import TemperatureDataset, TemperatureModel
from Model.funcs.funcs_lstm_multi import TemperatureDataset_multi, TemperatureModel_multi_full
from optuna.integration import PyTorchLightningPruningCallback
import random
import numpy as np
import yaml

forecast_vars=[ "press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50",     "rain", "wind_10", "wind_50","wind_dir_50_sin", "wind_dir_50_cos"]
storage="/home/alex/Dokumente/storage"
logs="/home/alex/Dokumente/lightning_logs"

#Seeds for reproducibility
pl.seed_everything(42)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

torch.set_float32_matmul_precision('medium')
def objective(trial):
    # Define the hyperparameters to optimize
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3,log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3,log=True)
    hidden_size = trial.suggest_categorical('hidden_size', [4,8,16, 32, 64,128])

    num_layers = trial.suggest_categorical('num_layers', [1, 2,4,6])
    batchsize=trial.suggest_int('batchsize', 32, 128, step=8)
    weight_intiliazier = trial.suggest_categorical('weight_initializer', [ "xavier","kaiming","normal"])
    window_size= trial.suggest_categorical('window_size', [24*7*4])
    # Initialize the model with the suggested hyperparameters
    training_data_path = storage+'/training/lstm_multi/train_' + forecast_var + "_" + str(window_size) + '.pt'
    val_data_path = storage+'/validation/lstm_multi/val_' + forecast_var + "_" + str(window_size) + '.pt'
    train_data = torch.load(training_data_path)
    val_data = torch.load(val_data_path)

    # Create data loaders for training and validation
    train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=False, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=batchsize, shuffle=False, num_workers=8)

    model = TemperatureModel_multi_full(hidden_size=hidden_size, learning_rate=learning_rate, weight_decay=weight_decay, num_layers=num_layers,
                                        weight_initializer=weight_intiliazier)
    logger = loggers.TensorBoardLogger(save_dir=logs+'/lstm_multi/' + forecast_var, name='lstm_optimierer')

    # Define the Lightning callbacks and trainer settings
    early_stopping = EarlyStopping('val_loss', patience=5, mode='min')
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val_loss')
    trainer = pl.Trainer(logger=logger, max_epochs=15, accelerator="auto", devices="auto",
                        callbacks=[early_stopping,pruning_callback], deterministic=True,enable_progress_bar=False)

    # Train the model using the pre-loaded train and validation loaders
    trainer.fit(model, train_loader, val_loader)



    return trainer.callback_metrics['val_loss'].item()

def export_best_params_and_model(forecast_var):
    # Erhalte die besten Parameter und speichere sie in einer Datei
    with open('output/lstm_multi/best_params_lstm_multi_' + forecast_var + '.yaml', 'w') as file:
        yaml.dump(best_params, file)

    # Trainiere das Modell mit den besten Parametern
    best_model = TemperatureModel_multi_full(hidden_size=best_params['hidden_size'], learning_rate=best_params['learning_rate'], weight_decay=best_params['weight_decay'],
                                  num_layers=best_params['num_layers'], weight_initializer=best_params['weight_initializer'])
    logger = loggers.TensorBoardLogger(save_dir=logs+'/lstm_multi/' + forecast_var, name='lstm_optimierer')
    trainer = pl.Trainer(logger=logger, max_epochs=20, accelerator="auto", devices="auto",
                         deterministic=True, enable_progress_bar=False)
    training_data_path = storage+'/training/lstm_multi/train_' + forecast_var + "_" + str(best_params['window_size']) + '.pt'
    val_data_path = storage+'/validation/lstm_multi/val_' + forecast_var + "_" + str(best_params['window_size']) + '.pt'
    train_data = torch.load(training_data_path)
    val_data = torch.load(val_data_path)

    # Create data loaders for training and validation
    train_loader = DataLoader(train_data, batch_size=best_params['batchsize'], shuffle=False, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=best_params['batchsize'], shuffle=False, num_workers=8)
    trainer.fit(best_model, train_loader, val_loader)

    # Speichere den Modellzustand des besten Modells
    torch.save(best_model.state_dict(), 'output/lstm_multi/models/best_model_state_'+forecast_var+'.pt')
    return



for forecast_var in forecast_vars:
    study = optuna.create_study(direction='minimize', storage='sqlite:///' + storage + '/database.db',
                                study_name="lstm_multi_test_" + forecast_var , load_if_exists=True)
    study.optimize(objective, n_trials=10)
    best_params = study.best_trial.params
    export_best_params_and_model(forecast_var)