import darts
import xarray as xr
import numpy as np
import optuna
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts.models import NaiveDrift
from darts.utils.statistics import plot_acf, check_seasonality
from darts import TimeSeries,concatenate
from darts.models import NaiveSeasonal
from darts.metrics import mape,smape,rmse
import datetime
import yaml
from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.models import ExponentialSmoothing, TBATS, AutoARIMA, Theta, NBEATSModel,NHiTSModel,TFTModel
import logging
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING) #disables the following output:


logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
data=xr.open_dataset('../Data/zusammengefasste_datei_2016-2022.nc').to_dataframe()[[ "press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50","rain", "wind_10", "wind_50","wind_dir_50_sin", "temp","wind_dir_50_cos"]]#[['index','temp']].to_dataframe()
#print(xr.open_dataset('../Data/zusammengefasste_datei_2016-2019.nc')[['index','temp']].isnull().any())
#print(data.columns)
#forecast_vars=["press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50","rain", "wind_10", "wind_50","wind_dir_50_sin","wind_dir_50_cos"]
forecast_vars=["temp"]
#print(data.loc[datetime.datetime(2016,12,31,23,0)])
#series= TimeSeries.from_xarray(data)
def main(forecast_var):
    series=TimeSeries.from_dataframe(data, value_cols=forecast_var,freq="h")
    #globals=TimeSeries.from_dataframe(data, value_cols="globalrcmp11",freq="h")
    #diffus=TimeSeries.from_dataframe(data, value_cols="diffuscmp11",freq="h")
    #humid=TimeSeries.from_dataframe(data, value_cols="humid",freq="h")
    #press_sl=TimeSeries.from_dataframe(data, value_cols="press_sl",freq="h")
    #rain=TimeSeries.from_dataframe(data, value_cols="rain",freq="h")
    #wind_10=TimeSeries.from_dataframe(data, value_cols="wind_10",freq="h")
    #wind_50=TimeSeries.from_dataframe(data, value_cols="wind_50",freq="h")
    #gust_10=TimeSeries.from_dataframe(data, value_cols="gust_10",freq="h")
    #gust_50=TimeSeries.from_dataframe(data, value_cols="gust_50",freq="h")
    #wind_dir_50_cos=TimeSeries.from_dataframe(data, value_cols="wind_dir_50_cos",freq="h")
    #wind_dir_50_sin=TimeSeries.from_dataframe(data, value_cols="wind_dir_50_sin",freq="h")
    #temp=TimeSeries.from_dataframe(data, value_cols="temp",freq="h")

    def past_covariates(main):
        cors = data[data.columns[:]].corr()[main]
        cors = cors[abs(cors) > 0]
        cors=abs(cors[cors<1])
        cors=cors.sort_values(ascending=False)
        num_offs=len(cors.index)
        offs=cors.index
        past_covariates = []
        data['Datum'] = pd.to_datetime(data.index)
        data['Tag_des_Jahres'] = np.float32(data['Datum'].dt.dayofyear)
        scaler = Scaler()
        #data["Tag_des_Jahres"] = scaler.fit_transform(data['Tag_des_Jahres'])
        past_covariates.append(TimeSeries.from_dataframe(data, value_cols="Tag_des_Jahres", freq="h"))
        past_covariates[0]=scaler.fit_transform(past_covariates[0])
        use_pasts=True
        for off in offs:
            print(off)
            scaler=Scaler()
            off=TimeSeries.from_dataframe(data, value_cols=off,freq="h")
            off=scaler.fit_transform(off)
            past_covariates.append(off)
        if num_offs >=3:
            past_covariates = concatenate([past_covariates[0], past_covariates[1], past_covariates[2],past_covariates[3]], axis=1)
        elif num_offs ==2:
            past_covariates = concatenate([past_covariates[0], past_covariates[1], past_covariates[2]], axis=1)
        elif num_offs ==1:
            past_covariates = concatenate([past_covariates[0], past_covariates[1]], axis=1)
        else:
            use_pasts=False
        return past_covariates,offs,use_pasts


    #covariates=[diffus,humid,press_sl,rain,wind_50,wind_10,gust_10,gust_50,temp,wind_dir_50_cos,wind_dir_50_sin]
    #print(covariates)
    #scalers=[str(x) for x in np.arange(0,len(covariates))]
    #gscaler=Scaler()
    #past_covariates=gscaler.fit_transform(globals)
    #past_covariates=[]
    #for cov,i in zip(covariates,scalers):

     #   i=Scaler()
       # cov=i.fit_transform(cov)
      #  #past_covariates=past_covariates.stack(cov)
        #past_covariates=concatenate([past_covariates,covs],axis=1)
        #past_covariates.append(cov)

    #print(diffus)
    #print(len(past_covariates))
    #covariaten=concatenate(covariates,axis=1)
    #past_covariates=concatenate([past_covariates[0],past_covariates[1],past_covariates[2],past_covariates[3],past_covariates[4],past_covariates[5],past_covariates[6],past_covariates[7],past_covariates[8],past_covariates[9]],axis=1)
    #past_covariates=concatenate([past_covariates[0],past_covariates[1],past_covariates[2],past_covariates[3],past_covariates[4],past_covariates[5],past_covariates[6],past_covariates[7]],axis=1)
    #past_covariates=concatenate([past_covariates[0],past_covariates[1],past_covariates[2]],axis=1)

    #train_glob,val_gob=globals.split_before(pd.Timestamp("2022-01-01 00:00") - datetime.timedelta(hours=672))

    past_covariates,offs,use_pasts=past_covariates(forecast_var)
    print(past_covariates)

    #encoders = {"datetime_attribute": {"past": ["month", "year"]}, "transformer": Scaler()}



    #model = NHiTSModel(input_chunk_length=672, output_chunk_length=24, random_state=42)
    #model= TFTModel(input_chunk_length=672,output_chunk_length=24,random_state=42,add_relative_index=True)

    preds=[]
    scaler = Scaler()
    series = scaler.fit_transform(series)

    train, predis = series.split_before(pd.Timestamp("2022-01-01 00:00") - datetime.timedelta(hours=672))
    train, test = train.split_before(0.7)

    test,val = test.split_before(0.1)

    # define objective function
    def objective(trial):
        # select input and output chunk lengths
        #in_len = trial.suggest_int("in_len",32,672)#trial.suggest_int("in_len",24 , 672*2,24)
        in_len= trial.suggest_categorical("in_len", [672])
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)


        out_len = 24#trial.suggest_int("out_len", 1, in_len-1)
        num_stacks=trial.suggest_int("num_stacks",2,5)
        num_blocks = trial.suggest_int("num_blocks", 1, 5)
        num_layers = trial.suggest_int("num_layers", 2, 5)
        layer_widths = trial.suggest_int("layer_widths", 256, 1024+256,256)
        batch_size = trial.suggest_int("batch_size", 32, 128)#trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        # Other hyperparameters
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        #include_year = trial.suggest_categorical("year", [True, False])

        # throughout training we'll monitor the validation loss for both pruning and early stopping
        pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
        early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=5, verbose=True)
        callbacks = [pruner, early_stopper]

        # detect if a GPU is available
        if torch.cuda.is_available():
            pl_trainer_kwargs = {
                "accelerator": "gpu",
                #"gpus": -1,
                #"auto_select_gpus": True,
                "callbacks": callbacks,
            }
            num_workers = 4
        else:
            pl_trainer_kwargs = {"callbacks": callbacks}
            num_workers = 0

        # optionally also add the (scaled) year value as a past covariate
        #if include_year:
            encoders =   {  #'cyclic': {'past': ['month']},
         #   'datetime_attribute': {'past': ['hour', 'dayofweek','dayofyear']},
            ''
            #'position': {'past': ['relative']},
          #  'transformer': Scaler()
}
        #else:
        #    encoders = None

        # reproducibility
        torch.manual_seed(42)

        # build the TCN model
        model = NHiTSModel(
            input_chunk_length=in_len,
            output_chunk_length=out_len,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            num_layers=num_layers,
            layer_widths=layer_widths,
            batch_size= batch_size,#64,
            n_epochs=15,
            dropout=dropout,
            optimizer_kwargs={"lr": lr, "weight_decay": weight_decay},
            #add_encoders=encoders,
            pl_trainer_kwargs=pl_trainer_kwargs,
            model_name="nhitsmodel"+forecast_var,
            force_reset=True,
            save_checkpoints=True,
        )
        #train, val = series.split_before(pd.Timestamp("2022-01-01 00:00") - datetime.timedelta(hours=in_len))
        # when validating during training, we can use a slightly longer validation
        # set which also contains the first input_chunk_length time steps
        #val = scaler.transform(series[-(24 + in_len) :])

        #print(train, val)
        # train the model
        if use_pasts==True:
            model.fit(
                series=train,
                val_series=val,
                num_loader_workers=num_workers,
                past_covariates=past_covariates,
                val_past_covariates=past_covariates,

            )
        else:
            model.fit(
                series=train,
                val_series=val,
                num_loader_workers=num_workers,

            )

        # reload best model over course of training
        model = NHiTSModel.load_from_checkpoint("nhitsmodel"+forecast_var)

        # Evaluate how good it is on the validation set, using sMAPE
        #preds = model.predict(series=train, n=len(test))
        for window, last_window in zip(range(in_len, len(test), 24),
                                       range(0, len(test) - in_len,24)):

            if last_window == 0:
                pred_tmp = model.predict(series=test[last_window:window], n=24, verbose=False,past_covariates=past_covariates)
                df = pred_tmp.pd_dataframe()

            else:
                pres = model.predict(series=test[last_window:window], n=24, verbose=False,past_covariates=past_covariates)
                fs = pres.pd_dataframe()
                df = pd.concat([df, fs])

        preds = TimeSeries.from_dataframe(df)
        smapes = rmse(test, preds, n_jobs=-1, verbose=True)
        smape_val = np.mean(smapes)

        return smape_val if smape_val != np.nan else float("inf")


    # for convenience, print some optimization trials information
    def print_callback(study, trial):
        print(f"Current value: {trial.value}, Current params: {trial.params}")
        print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


    # optimize hyperparameters by minimizing the sMAPE on the validation set
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10, callbacks=[print_callback])
    best_params = study.best_trial.params

    with open('output/nhits/best_params_nhits_' + forecast_var + '.yaml', 'w') as file:
        yaml.dump(best_params, file)
    #.load_from_checkpoint("nhitsmodel")
    #model.fit([train], epochs=30, verbose=True,past_covariates=past_covariates)
    print(best_params["in_len"])

    #if best_params["include_year"]:
     #   encoders = {'cyclic': {'past': ['month']},
       #             'datetime_attribute': {'past': ['hour', 'dayofweek', 'dayofyear']},
      #              'position': {'past': ['relative']},
                    # 'custom': {'past': [lambda idx: (idx.year - 1950) / 50]},
        #            'transformer': Scaler()
         #           }
    #else:
     #   encoders = None
    #in_len=24*47
    #model = NHiTSModel.load_from_checkpoint("nhitsmodel_opti")
    torch.manual_seed(42)
    model = NHiTSModel(
        input_chunk_length=best_params["in_len"],
        output_chunk_length=24,
        num_stacks=best_params["num_stacks"],
        num_blocks=best_params["num_blocks"],
        num_layers=best_params["num_layers"],
        layer_widths=best_params["layer_widths"],
        batch_size= best_params["batch_size"],#64,
        #n_epochs=20,
        #add_encoders=encoders,
        dropout=best_params["dropout"],
        optimizer_kwargs={"lr": best_params["lr"], "weight_decay": best_params["weight_decay"]},
        model_name="nhitsmodel",
    )
    #train, val = preds.split_before(pd.Timestamp("2022-01-01 00:00")-datetime.timedelta(hours=best_params["in_len"]))
    if use_pasts==True:
        model.fit( epochs=15, verbose=False,
                  series=train,
                  val_series=val,
                  past_covariates=past_covariates,
                  val_past_covariates=past_covariates,

                  )
    else:
        model.fit(series=train,
                  val_series=val, epochs=15, verbose=False)
    hourly_range = pd.date_range(start="2022-01-01 00:00", end="2022-12-31 23:00", freq='H')

    rest, predis = predis.split_before(pd.Timestamp("2022-01-01 00:00")-datetime.timedelta(hours=best_params["in_len"]))
    for window, last_window in zip(range(best_params["in_len"], len(predis), 24),
                                   range(0, len(predis) - best_params["in_len"],
                                         24)):

        if last_window==0:
            pred_tmp = model.predict(series=predis[last_window:window], n=24,verbose=False,past_covariates=past_covariates)
            #print(pred_tmp.univariate_values.flatten())
            pred_tmp = (scaler.inverse_transform(pred_tmp))
            df=pred_tmp.pd_dataframe()
            print(df)

        else:
            pres= model.predict(series=predis[last_window:window], n=24,verbose=False,past_covariates=past_covariates)
            #pred_tmp.append_values(pres)
            pres = (scaler.inverse_transform(pres))
            fs = pres.pd_dataframe()
            df=pd.concat([df, fs])

        #preds = (scaler.inverse_transform(pred_tmp))
        #print(pred_tmp)
        #preds.append(np.array(pred_tmp.values).flatten())

    #print(np.array(preds).flatten())
    #preds=np.array(preds).flatten
    #datas = pd.Series(preds, index=hourly_range)
    print(df.tail())
    print(predis.tail())
    print(len(df))
    preds=TimeSeries.from_dataframe(df)
    preds.plot(label="Forecast")
    val=scaler.inverse_transform(val)
    val.plot(label="Obs")
    #plt.show()
    file= "../Visualistion/best_nhit_"+forecast_var+".nc"
    output=preds.pd_dataframe().to_xarray().to_netcdf(file)
    return

for forecast_var in forecast_vars:
    main(forecast_var)






