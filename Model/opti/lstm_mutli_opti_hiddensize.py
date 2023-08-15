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
forecast_vars =[ 'temp']
#forecast_vars=[ "press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50",     "rain", "wind_10", "wind_50","wind_dir_50_sin", "wind_dir_50_cos"]
storage="/home/alex/Dokumente/storage"
logs="/home/alex/Dokumente/lightning_logs"
full='../../Data/zusammengefasste_datei_2016-2021.nc'
file_path =full # Replace with the actual path to your NetCDF file
# Setzen Sie die Zufallssaat für die GPU
# Setze den Random Seed für PyTorch
pl.seed_everything(42)

# Setze den Random Seed für torch
torch.manual_seed(42)

# Setze den Random Seed für random
random.seed(42)

# Setze den Random Seed für numpy
np.random.seed(42)

torch.set_float32_matmul_precision('medium')
def objective(trial):
    # Define the hyperparameters to optimize
    learning_rate = 0.0001#trial.suggest_float('learning_rate', 1e-5, 1e-3,log=True)
    weight_decay =1e-5 #trial.suggest_float('weight_decay', 1e-5, 1e-3,log=True)
    hidden_size = trial.suggest_categorical('hidden_size', [4 ,64,128,256,512,1024])
    #optimizer  = trial.suggest_categorical('optimizer', ["Adam","AdamW"])
    #dropout = trial.suggest_categorical('dropout', [0,0.2,0.5])
    num_layers = trial.suggest_categorical('num_layers', [1, 2,4,6])
    #batchsize = trial.suggest_categorical('batchsize', [1,2,3,6,12,24])
    batchsize=60#trial.suggest_int('batchsize', 32, 128, step=8)
    weight_intiliazier = "kaiming"#trial.suggest_categorical('weight_initializer', [ "xavier","kaiming","normal"])
    window_size= 24#trial.suggest_categorical('window_size', [24*7*4])
    # Initialize the model with the suggested hyperparameters
    #training_data_path = storage+'/training/lstm_multi/train_' + forecast_var + "_" + str(window_size) + '.pt'
    #val_data_path = storage+'/validation/lstm_multi/val_' + forecast_var + "_" + str(window_size) + '.pt'
    corvars=["temp","press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50",     "rain", "wind_10", "wind_50","wind_dir_50_sin", "wind_dir_50_cos","taupunkt","Taupunkt3h","press3h","rainsum3h","temp3h","gradwind","rain_event"]
    dataset = TemperatureDataset_multi(file_path, forecast_horizont=24, window_size=window_size,
                                       forecast_var=forecast_var, cor_vars=corvars)
    train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)
    #train_data = torch.load(training_data_path)
    #val_data = torch.load(val_data_path)

    # Create data loaders for training and validation
    train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=False, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=batchsize, shuffle=False, num_workers=8)

    model = TemperatureModel_multi_full(hidden_size=hidden_size, learning_rate=learning_rate, weight_decay=weight_decay, num_layers=num_layers,
                                        weight_initializer=weight_intiliazier,numvars=len(corvars))
    logger = loggers.TensorBoardLogger(save_dir=logs+'/lstm_multi/' + forecast_var, name='lstm_optimierer2')

    # Define the Lightning callbacks and trainer settings
    early_stopping = EarlyStopping('val_loss', patience=5, mode='min')
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val_loss')
    trainer = pl.Trainer(logger=logger, max_epochs=15, accelerator="auto", devices="auto",
                        callbacks=[early_stopping,pruning_callback], deterministic=True,enable_progress_bar=False)

    # Train the model using the pre-loaded train and validation loaders
    trainer.fit(model, train_loader, val_loader)
    #trial.report(trainer.callback_metrics['val_loss'], epoch=trainer.current_epoch)

    # Handle pruning based on the reported value
    #if trial.should_prune():
     #   raise optuna.exceptions.TrialPruned()

    # Return the performance metric you want to optimize (e.g., validation loss)
    return trainer.callback_metrics['val_loss'].item()

def export_best_params_and_model(forecast_var):
    # Erhalte die besten Parameter und speichere sie in einer Datei
    with open('output/lstm_multi/hidden/best_params_lstm_multi_' + forecast_var + '.yaml', 'w') as file:
        yaml.dump(best_params, file)


    return



# Erhalte die besten Parameter und speichere sie in einer Datei

for forecast_var in forecast_vars:
    study = optuna.create_study(direction='minimize', storage='sqlite:///' + storage + '/database_2.db',
                                study_name="lstm_multi_test_scoy" + forecast_var , load_if_exists=True)
    study.optimize(objective, n_trials=10)
    best_params = study.best_trial.params
    export_best_params_and_model(forecast_var)