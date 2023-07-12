import optuna_dashboard,optuna
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
#from funcs.funcs_lstm_single import TemperatureDataset, TemperatureModel
from Model.funcs.funcs_lstm_multi import TemperatureDataset_multi, TemperatureModel_multi_full
from Model.funcs.funcs_tft import TFT_Modell
from optuna.integration import PyTorchLightningPruningCallback
import random
import numpy as np
import yaml
forecast_var = 'temp'
# Setzen Sie die Zufallssaat für die GPU
# Setze den Random Seed für PyTorch
pl.seed_everything(42)

# Setze den Random Seed für torch
torch.manual_seed(42)

# Setze den Random Seed für random
random.seed(42)

# Setze den Random Seed für numpy
np.random.seed(42)

#forecast_vars=[ "press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50",     "rain", "wind_10", "wind_50","wind_dir_50_sin", "wind_dir_50_cos"]
forecast_vars=["temp"]
storage="/home/alex/Dokumente/storage/"
logs="/home/alex/Dokumente/lightning_logs/"

torch.set_float32_matmul_precision('medium')
def objective(trial):
    hidden_dim = trial.suggest_categorical('hidden_dim',[8,16])
    num_layers = trial.suggest_categorical('num_layers', [1, 2, 3])
    num_heads = trial.suggest_categorical('num_heads',[2,  4, 8,])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    #batchsize = trial.suggest_categorical('batchsize', [24])
    batchsize = trial.suggest_int('batchsize', 24, 64, step=8)
    weight_initializer = trial.suggest_categorical('weight_intiliazier', [ "xavier","kaiming","normal"])
    window_size= trial.suggest_categorical('window_size', [24*7*4])
    input_dim = 12  # Anzahl der meteorologischen Parameter
    output_dim = 1  # Vorhersage für einen Wert
    # Initialize the model with the suggested hyperparameters
    training_data_path = storage+'training/lstm_multi/train_' + forecast_var + "_" + str(window_size) + '.pt'
    val_data_path = storage+'validation/lstm_multi/val_' + forecast_var + "_" + str(window_size) + '.pt'
    train_data = torch.load(training_data_path)
    val_data = torch.load(val_data_path)

    # Create data loaders for training and validation
    train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=False, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=batchsize, shuffle=False, num_workers=8)

    model = TFT_Modell(input_dim=input_dim, output_dim=output_dim,hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout,
                       learning_rate=learning_rate, weight_decay=weight_decay,weight_initializer=weight_initializer,window_size=window_size)
    logger = loggers.TensorBoardLogger(save_dir='../lightning_logs/tft/' + forecast_var, name='tft_optimierer')

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
    with open('output/tft/best_params_tft' + forecast_var + '.yaml', 'w') as file:
        yaml.dump(best_params, file)

    # Trainiere das Modell mit den besten Parametern
    best_model = TFT_Modell(input_dim=best_params["input_dim"], output_dim=best_params["output_dim"],hidden_dim=best_params["hidden_dim"], num_layers=best_params["num_layers"], num_heads=best_params["num_heads"], dropout=best_params["dropout"],
                       learning_rate=best_params["learning_rate"], weight_decay=best_params["weight_decay"],weight_initializer=best_params["weight_initializer"],window_size=best_params["window_size"])
    logger = loggers.TensorBoardLogger(save_dir=logs+'/tft/' + forecast_var, name='tft')
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
    torch.save(best_model.state_dict(), 'output/tft/models/best_model_state_'+forecast_var+'.pt')
    return



# Erhalte die besten Parameter und speichere sie in einer Datei

for forecast_var in forecast_vars:
    study = optuna.create_study(direction='minimize', storage='sqlite:///' + storage + '/database.db',
                                study_name="tft_1" + forecast_var , load_if_exists=True)
    study.optimize(objective, n_trials=10)
    best_params = study.best_trial.params
    export_best_params_and_model(forecast_var)