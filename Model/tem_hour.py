import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from funcs.funcs_lstm_single  import TemperatureModel
import random
import numpy as np
forecast_var = 'temp'

pl.seed_everything(42)
torch.set_float32_matmul_precision('medium')

torch.manual_seed(42)

random.seed(42)


np.random.seed(42)




#Hyper Parameters:
window_size = 24*7*4#168
learning_rate =0.00028321349862445627#0.00005#
weight_decay =6.814701853104705e-05# 0.0001#
hidden_size =64#32
optimizer= "Adam"
num_layers=2#1
dropout=0#0.5
weight_initializer="kaiming"

training_data_path = 'opti/storage/training_data_lstm_single_train_' + forecast_var + "_" + str(window_size) + '.pt'
val_data_path = 'opti/storage/training_data_lstm_single_val_' + forecast_var + "_" + str(window_size) + '.pt'
train_data = torch.load(training_data_path)
val_data = torch.load(val_data_path)

    # Create data loaders for training and validation
train_loader = DataLoader(train_data, batch_size=24, shuffle=False, num_workers=8)
val_loader = DataLoader(val_data, batch_size=24, shuffle=False, num_workers=8)



# Initialize the model with the suggested hyperparameters
model = TemperatureModel(hidden_size=hidden_size, learning_rate=learning_rate, weight_decay=weight_decay,optimizer=optimizer,num_layers=num_layers,dropout=dropout)



logger = loggers.TensorBoardLogger(save_dir='lightning_logs/lstm_uni/'+forecast_var, name='lstm_unoptimiert')
early_stopping = EarlyStopping('val_loss', patience=5,mode='min')
trainer = pl.Trainer(logger=logger,max_epochs=50, accelerator="auto",devices="auto",callbacks=[early_stopping],deterministic=True)#,val_check_interval=0.5) #log_every_n_steps=2,
trainer.fit(model, train_loader,val_loader)

  # Verwende hier den entsprechenden Dataloader (z.B. val_loader)
torch.save(model.state_dict(), 'output/lstm_uni/'+forecast_var+'optimierter.pth')