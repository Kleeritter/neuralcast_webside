import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
#from funcs.funcs_lstm_multi import TemperatureDataset_multi, TemperatureModel_multi_full
from Model.funcs.funcs_lstm_single import TemperatureDataset, TemperatureModel
from Visualistion.createframes import forecast_lstm_multi,forecast_lstm_uni
import yaml
from  tqdm  import tqdm

full='../Data/zusammengefasste_datei_2016-2021.nc'
file_path =full # Replace with the actual path to your NetCDF file
logs ="/home/alex/Dokumente/lightning_logs"

def train(forecast_var,forecast_horizont,window_size,transfer_learning=False,corvars=[ "temp","press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50",     "rain", "wind_10", "wind_50","wind_dir_50_sin", "wind_dir_50_cos","taupunkt","Taupunkt3h","press3h","rainsum3h","temp3h","gradwind","rain_event"],Ruthe=False):
    # Lade die Hyperparameter
    with open('../opti/output/lstm_single/best_params_lstm_single_'+forecast_var+'.yaml') as file:
        best_params = yaml.load(file, Loader=yaml.FullLoader)

    #cor_vars=["temp","press"]
   # dataset = TemperatureDataset_multi(file_path, forecast_horizont=forecast_horizont, window_size=window_size,
    #                                   forecast_var=forecast_var,cor_vars=corvars)
    dataset =TemperatureDataset(file_path, forecast_horizont=forecast_horizont, window_size=window_size,
                                       forecast_var=forecast_var)
    train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)
    train_loader = DataLoader(train_data, batch_size=best_params["batchsize"], shuffle=False, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=best_params["batchsize"], shuffle=False, num_workers=8)
    # Trainiere das Modell mit den besten Parametern

   # model = TemperatureModel_multi_full(hidden_size=best_params['hidden_size'], learning_rate=best_params['learning_rate'], weight_decay=best_params['weight_decay'],
    #                                num_layers=best_params['num_layers'], weight_initializer=best_params['weight_initializer'],forecast_horizont=forecast_horizont, window_size=window_size,numvars=len(corvars))

    model = TemperatureModel(hidden_size=best_params['hidden_size'], learning_rate=best_params['learning_rate'], weight_decay=best_params['weight_decay'],num_layers=best_params['num_layers'],
                             weight_intiliazier=best_params['weight_intiliazier'],forecast_horizont=forecast_horizont, window_size=window_size)
    logger = loggers.TensorBoardLogger(save_dir=logs+'/lstm_single/' + forecast_var, name='lstm_optimierer')
    trainer = pl.Trainer(logger=logger, max_epochs=20, accelerator="auto", devices="auto",
                         deterministic=True, enable_progress_bar=True, callbacks=[EarlyStopping(monitor='val_loss', patience=5, mode='min')])



    # Create data loaders for training and validation
    train_loader = DataLoader(train_data, batch_size=best_params['batchsize'], shuffle=False, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=best_params['batchsize'], shuffle=False, num_workers=8)
    trainer.fit(model, train_loader, val_loader)

    # Speichere den Modellzustand des besten Modells
    if Ruthe:
        torch.save(model.state_dict(), '../Ruthe/lstm_single/models/best_model_state_'+forecast_var+'_'+str(window_size)+'_'+str(forecast_horizont)+'.pt')
    else:
        torch.save(model.state_dict(), 'lstm_single/models/best_model_state_'+forecast_var+'_'+str(window_size)+'_'+str(forecast_horizont)+'.pt')
    return

if __name__ == '__main__':
    forecast_vars=["temp","humid","wind_10","gust_10","rain","globalrcmp11"]#[ "temp","press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50",     "rain", "wind_10", "wind_50","wind_dir_50_sin", "wind_dir_50_cos"]
    for forecast_var in tqdm(forecast_vars):
        for window_size in [24]:#[2*7*24,7*24,6*24,5*24,4*24,3*24,2*24,24,12,6,3]):#,8*7*24,4*7*24,2*7*24,7*24,6*24,4*24,3*24,2*24,24,12,6,3]):
            for forecast_horizont in [24]:#2,4,6,12,15,18,24,32,48,60,72,84,96,192]:
                #if window_size!=5*24 :
                   # train(forecast_var,forecast_horizont,window_size,transfer_learning=True)
                #else:
                 #   train(forecast_var,forecast_horizont,window_size,transfer_learning=False)
                #train(forecast_var, forecast_horizont, window_size, transfer_learning=False)
                model_path='timetest/lstm_single/models/best_model_state_'+forecast_var+'_'+str(window_size)+'_'+str(forecast_horizont)+'.pt'
                params_folder='opti/output/lstm_single/best_params_lstm_single_'+forecast_var+'.yaml'
                varlist = [forecast_var]
                #output_file='timetest/lstm_multi/output/'+forecast_var+'/timetest_lstm_multi'+forecast_var+'_'+str(window_size)+'_'+str(forecast_horizont)+'.nc'
                output_file='Ruthe/lstm_single/output/all/timetest_lstm_single'+forecast_var+'_'+str(window_size)+'_'+str(forecast_horizont)+'.nc'
                #forecast_lstm_multi(var_list=varlist, model_folder=model_path, params_folder=params_folder,forecast_horizont=forecast_horizont,window_size=window_size,output_file=output_file,corvars=[ "temp","press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50",     "rain", "wind_10", "wind_50","wind_dir_50_sin", "wind_dir_50_cos","taupunkt","Taupunkt3h","press3h","rainsum3h","temp3h","gradwind","rain_event"])
                forecast_lstm_uni(model_folder=model_path, params_folder=params_folder,forecast_horizont=forecast_horizont,window_size=window_size,output_file=output_file,var_list=varlist,ruthe=True,forecast_year=2020)