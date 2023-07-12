import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import xarray as xr
import pandas as pd
import numpy as np
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss, RMSE,MASE
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
torch.set_float32_matmul_precision('medium')
import lightning.pytorch as pl
from sklearn.preprocessing import MinMaxScaler
from Model.funcs.visualer_funcs import load_hyperparameters
import warnings

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

# Daten vorbereiten
# Nehmen wir an, Sie haben Ihre Daten in einem Pandas DataFrame namens "data" organisiert

# Definieren Sie die Spalten, die als Features und als Ziel verwendet werden sollen
target_col = "temp"
feature_cols =[  'humid', 'temp', 'press_sl',        'wind_10', 'wind_50', 'gust_10', 'gust_50', 'rain', 'diffuscmp11', 'globalrcmp11', 'wind_dir_50_sin', 'wind_dir_50_cos']  # Liste der meteorologischen Parameter

data=xr.open_dataset('../Data/zusammengefasste_datei_2016-2019.nc').to_dataframe()
data["timestep"] = np.arange(0, len(data.index.tolist()))#pd.to_numeric(data.index)- pd.to_numeric(data.index).min()
data["const"]=np.ones(len(data))
data=data.reset_index()
data=data.drop(columns="index")
#data=data[["temp","humid","timestep","const"]]
#print(data.columns)
for column in feature_cols:
    values = data[column].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    param_path ="../Data/params_for_normal.yaml" #'/home/alex/PycharmProjects/nerualcaster/Data/params_for_normal.yaml'  # "../../Data/params_for_normal.yaml"
    params = load_hyperparameters(param_path)
    mins = params["Min_" + column]
    maxs = params["Max_" + column]
    train_values = [mins, maxs]
    X_train_minmax = scaler.fit_transform(np.array(train_values).reshape(-1, 1))
    # self.data = scaler.transform([[x] for x in self.data]).flatten()
    scaled_values = scaler.transform(values)
    data[column] = scaled_values.flatten()
data=data[[  'humid', 'temp', 'press_sl',        'wind_10', 'wind_50', 'gust_10', 'gust_50', 'rain', 'diffuscmp11', 'globalrcmp11', 'wind_dir_50_sin', 'wind_dir_50_cos',"timestep","const"]]


max_prediction_length = 24
max_encoder_length = 672
training_cutoff = data["timestep"].max() - max_prediction_length
# Erstellen Sie ein TimeSeriesDataSet-Objekt
training = TimeSeriesDataSet(
    data=data[lambda x: x.timestep <= training_cutoff],
    target=target_col,
    group_ids=["const"],  # Annahme: Spalte mit Zeitstempeln heißt "Timestamp"
    time_idx="timestep",
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=feature_cols,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)


validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)



# create dataloaders for model
batch_size = 32  # set this between 32 and 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=16)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=16)

# calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
baseline_predictions = Baseline().predict(val_dataloader, return_y=True)
MAE()(baseline_predictions.output, baseline_predictions.y)

pl.seed_everything(42)
# configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=10,
    accelerator="auto",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    limit_train_batches=50,  # coment in for training, running valiation every 30 batches
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=8,
    loss=QuantileLoss(),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    optimizer="Ranger",
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

#trainer.fit(
 #   tft,
  #  train_dataloaders=train_dataloader,
   # val_dataloaders=val_dataloader,
#)

#torch.save(tft.state_dict(), 'output/ttft/'+target_col+'.ckpt')

import pickle

from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

# create study
study = optimize_hyperparameters(
    train_dataloader,
    val_dataloader,
    model_path="optuna_test",
    n_trials=10,
    max_epochs=50,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(8, 128),
    hidden_continuous_size_range=(8, 128),
    attention_head_size_range=(1, 4),
    learning_rate_range=(0.001, 0.1),
    dropout_range=(0.1, 0.3),
    trainer_kwargs=dict(limit_train_batches=30),
    reduce_on_plateau_patience=4,
    use_learning_rate_finder=False,
    log_dir="lightning_logs"# use Optuna to find ideal learning rate or use in-built learning rate finder
)

# save study results - also we can resume tuning at a later point in time
with open("test_study.pkl", "wb") as fout:
    pickle.dump(study, fout)

# show best hyperparameters
print(study.best_trial.params)