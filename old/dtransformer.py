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
import yaml
import matplotlib.pyplot as plt
from darts.models import NaiveDrift
from darts.utils.statistics import plot_acf, check_seasonality
from darts import TimeSeries,concatenate
from darts.models import NaiveSeasonal
from darts.metrics import mape,smape,rmse
import datetime
from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.models import ExponentialSmoothing, TBATS, AutoARIMA, Theta, NBEATSModel,NHiTSModel,TFTModel
data=xr.open_dataset('../Data/zusammengefasste_datei_2016-2022.nc').to_dataframe()[[ "press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50","rain", "wind_10", "wind_50","wind_dir_50_sin", "temp","wind_dir_50_cos"]]#[['index','temp']].to_dataframe()

#forecast_var="temp"
#forecast_vars=["press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50","rain", "wind_10", "wind_50","wind_dir_50_sin","wind_dir_50_cos"]
forecast_vars=["temp"]
def main(forecast_var):
    series=TimeSeries.from_dataframe(data, value_cols=forecast_var,freq="h")


    def past_covariates(main):
        cors = data[data.columns[:]].corr()[main]
        cors = cors[abs(cors) > 0]
        cors=abs(cors[cors<1])
        cors=cors.sort_values(ascending=False)
        num_offs=len(cors.index)
        offs=cors.index
        past_covariates = []
        use_pasts=True
        for off in offs:
            print(off)
            scaler=Scaler()
            off=TimeSeries.from_dataframe(data, value_cols=off,freq="h")
            off=scaler.fit_transform(off)
            past_covariates.append(off)
        if num_offs >=3:
            past_covariates = concatenate([past_covariates[0], past_covariates[1], past_covariates[2]], axis=1)
        elif num_offs ==2:
            past_covariates = concatenate([past_covariates[0], past_covariates[1]], axis=1)
        elif num_offs ==1:
            past_covariates = concatenate([past_covariates[0]], axis=1)
        else:
            use_pasts=False
        return past_covariates,offs,use_pasts


    past_covariates,offs,use_pasts=past_covariates(forecast_var)



    preds=[]
    scaler = Scaler()
    series = scaler.fit_transform(series)


    def objective(trial):
        # select input and output chunk lengths
        in_len = trial.suggest_int("in_len",24 , 672*2,24)
        out_len = 24#trial.suggest_int("out_len", 1, in_len-1)
        hidden_size=trial.suggest_int("hidden_size",8,32,4)
        lstm_layers=trial.suggest_int("lstm_layers",1,3)
        num_attention_heads=trial.suggest_int("num_attention_heads",2,8,2)
        # Other hyperparameters
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        #include_year = trial.suggest_categorical("year", [False, True])

        # throughout training we'll monitor the validation loss for both pruning and early stopping
        pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
        early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=3, verbose=True)
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
       # if include_year:
            encoders = {"datetime_attribute": {"past": ["year"]},
                        "transformer": Scaler()}
        #else:
         #   encoders = None

        # reproducibility
        torch.manual_seed(42)

        # build the TCN model
        model = TFTModel(
            input_chunk_length=in_len,
            output_chunk_length=out_len,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            num_attention_heads=num_attention_heads,
            batch_size=64,
            n_epochs=1,
            dropout=dropout,
            add_relative_index=True,
            optimizer_kwargs={"lr": lr},
            #add_encoders=encoders,
            pl_trainer_kwargs=pl_trainer_kwargs,
            model_name="tftmodell"+forecast_var,
            force_reset=True,
            save_checkpoints=True,
            likelihood=None, loss_fn=torch.nn.MSELoss()
        )

        # when validating during training, we can use a slightly longer validation
        # set which also contains the first input_chunk_length time steps
        #model_val_set = scaler.transform(series[-(24 + in_len) :])
        train, val = series.split_before(pd.Timestamp("2022-01-01 00:00") - datetime.timedelta(hours=in_len))
        # train the model
        model.fit(
            series=train,
            val_series=val,
            num_loader_workers=num_workers,
            past_covariates=past_covariates,
            val_past_covariates=past_covariates,

        )

        # reload best model over course of training
        model = TFTModel.load_from_checkpoint("tftmodell"+forecast_var)

        # Evaluate how good it is on the validation set, using sMAPE
        preds = model.predict(series=train, n=24,num_samples=1)
        smapes = rmse(val, preds, n_jobs=-1, verbose=True)
        smape_val = np.mean(smapes)

        return smape_val if smape_val != np.nan else float("inf")


    # for convenience, print some optimization trials information
    def print_callback(study, trial):
        print(f"Current value: {trial.value}, Current params: {trial.params}")
        print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


    # optimize hyperparameters by minimizing the sMAPE on the validation set
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1, callbacks=[print_callback])
    best_params = study.best_trial.params

    with open('output/tft_dart/best_params_tft_' + forecast_var + '.yaml', 'w') as file:
        yaml.dump(best_params, file)

    #.load_from_checkpoint("nhitsmodel")
    #model.fit([train], epochs=30, verbose=True,past_covariates=past_covariates)
    print(best_params["in_len"])
    #in_len=24*47
    #model = NHiTSModel.load_from_checkpoint("nhitsmodel_opti")
    model = TFTModel(
        input_chunk_length=best_params["in_len"],
        output_chunk_length=24,
        hidden_size=best_params["hidden_size"],
        lstm_layers=best_params["lstm_layers"],
        num_attention_heads=best_params["num_attention_heads"],
        batch_size=64,
        n_epochs=5,
        dropout=best_params["dropout"],
        add_relative_index=True,
        optimizer_kwargs={"lr": best_params["lr"]},
        likelihood=None, loss_fn=torch.nn.MSELoss(),
    )
    train, val = series.split_before(pd.Timestamp("2022-01-01 00:00")-datetime.timedelta(hours=best_params["in_len"]))

    model.fit([train], epochs=5, verbose=True,past_covariates=past_covariates)
    hourly_range = pd.date_range(start="2022-01-01 00:00", end="2022-12-31 23:00", freq='H')

    trains, vals = series.split_before(pd.Timestamp("2022-01-01 00:00")-datetime.timedelta(hours=best_params["in_len"]))
    for window, last_window in zip(range(best_params["in_len"], len(vals), 24),
                                   range(0, len(vals) - best_params["in_len"],
                                         24)):

        if last_window==0:
            pred_tmp = model.predict(series=vals[last_window:window], n=24,verbose=False,num_samples=1)
            #print(pred_tmp.univariate_values.flatten())
            pred_tmp = (scaler.inverse_transform(pred_tmp))
            df=pred_tmp.pd_dataframe()
            print(df)

        else:
            pres= model.predict(series=vals[last_window:window], n=24,verbose=False)
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
    print(len(df))
    preds=TimeSeries.from_dataframe(df)
    preds.plot(label="Forecast")
    val=scaler.inverse_transform(val)
    val.plot(label="Obs")
    plt.show()
    file= "../Visualistion/best_transformer_"+forecast_var+".nc"
    output=preds.pd_dataframe().to_xarray().to_netcdf(file)
    return

for forecast_var in forecast_vars:
    main(forecast_var)




