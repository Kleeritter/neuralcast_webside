import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
# from funcs.funcs_lstm_multi import TemperatureDataset_multi, TemperatureModel_multi_full
from Model.funcs.funcs_lstm_single import TemperatureDataset, TemperatureModel
from Visualistion.createframes import forecast_lstm_multi, forecast_lstm_uni
import yaml
from tqdm import tqdm

# Define file paths and directories
full = '../Data/zusammengefasste_datei_2016-2021.nc'
file_path = full  # Replace with the actual path to your NetCDF file
logs = "/home/alex/Dokumente/lightning_logs"


def train(forecast_var, forecast_horizont, window_size, transfer_learning=False,
          corvars=["temp", "press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50", "rain", "wind_10",
                   "wind_50", "wind_dir_50_sin", "wind_dir_50_cos", "taupunkt", "Taupunkt3h", "press3h", "rainsum3h",
                   "temp3h", "gradwind", "rain_event"], Ruthe=False):
    # Load best hyperparameters from file
    with open('opti/output/lstm_single/best_params_lstm_single_' + forecast_var + '.yaml') as file:
        best_params = yaml.load(file, Loader=yaml.FullLoader)

    # Create the dataset using TemperatureDataset class
    dataset = TemperatureDataset(file_path, forecast_horizont=forecast_horizont, window_size=window_size,
                                 forecast_var=forecast_var)

    # Split dataset into training and validation sets
    train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)

    # Create DataLoader instances for training and validation
    train_loader = DataLoader(train_data, batch_size=best_params["batchsize"], shuffle=False, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=best_params["batchsize"], shuffle=False, num_workers=8)

    # Create the model
    model = TemperatureModel(hidden_size=best_params['hidden_size'], learning_rate=best_params['learning_rate'],
                             weight_decay=best_params['weight_decay'], num_layers=best_params['num_layers'],
                             weight_intiliazier=best_params['weight_intiliazier'], forecast_horizont=forecast_horizont,
                             window_size=window_size)

    # Set up TensorBoardLogger and trainer
    logger = loggers.TensorBoardLogger(save_dir=logs + '/lstm_single/' + forecast_var, name='lstm_optimierer')
    trainer = pl.Trainer(logger=logger, max_epochs=20, accelerator="auto", devices="auto", deterministic=True,
                         enable_progress_bar=True,
                         callbacks=[EarlyStopping(monitor='val_loss', patience=5, mode='min')])

    # Fit the model on the training data
    trainer.fit(model, train_loader, val_loader)

    # Save the trained model
    if Ruthe:
        torch.save(model.state_dict(),
                   '../Ruthe/lstm_single/models/best_model_state_' + forecast_var + '_' + str(window_size) + '_' + str(
                       forecast_horizont) + '.pt')
    else:
        torch.save(model.state_dict(),
                   'lstm_single/models/best_model_state_' + forecast_var + '_' + str(window_size) + '_' + str(
                       forecast_horizont) + '.pt')
    return


if __name__ == '__main__':
    forecast_vars = ["temp", "humid", "wind_10", "gust_10", "rain", "globalrcmp11"]
    for forecast_var in tqdm(forecast_vars):
        for window_size in [24]:
            for forecast_horizont in [24]:
               # train(forecast_var, forecast_horizont, window_size, transfer_learning=False,Ruthe=True)

                model_path = 'timetest/lstm_single/models/best_model_state_' + forecast_var + '_' + str(
                    window_size) + '_' + str(forecast_horizont) + '.pt'
                params_folder = 'opti/output/lstm_single/best_params_lstm_single_' + forecast_var + '.yaml'
                varlist = [forecast_var]
                output_file = 'Ruthe/lstm_single/output/all/timetest_lstm_single' + forecast_var + '_' + str(
                    window_size) + '_' + str(forecast_horizont) + '.nc'

                # Generate forecasts using the trained model
                forecast_lstm_uni(model_folder=model_path, params_folder=params_folder,
                                  forecast_horizont=forecast_horizont, window_size=window_size, output_file=output_file,
                                  var_list=varlist, ruthe=True, forecast_year=2020)
