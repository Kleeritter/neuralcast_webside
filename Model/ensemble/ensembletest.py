from darts.models import RegressionEnsembleModel, NaiveDrift, NaiveSeasonal,RNNModel
from darts.datasets import AirPassengersDataset
from matplotlib import pyplot as plt
import pytorch_lightning
import torch
import yaml
from Model.funcs.funcs_lstm_multi import TemperatureModel_multi_full
from Model.funcs.funcs_lstm_single import TemperatureModel
forecast_var="temp"
forecast_horizont=24
window_size=24
with open('../opti/output/lstm_multi/best_params_lstm_multi_' + forecast_var + '.yaml') as file:
    best_params = yaml.load(file, Loader=yaml.FullLoader)
lstm_multi = TemperatureModel_multi_full(hidden_size=best_params['hidden_size'],
                                    learning_rate=best_params['learning_rate'],
                                    weight_decay=best_params['weight_decay'],
                                    num_layers=best_params['num_layers'],
                                    weight_initializer=best_params['weight_initializer'],
                                    forecast_horizont=forecast_horizont, window_size=window_size)
checkpoint = torch.load('../timetest/lstm_multi/models/best_model_state_' + forecast_var + '_' + str( 24) + '_' + str(
    forecast_horizont) + '.pt')
lstm_multi.load_state_dict(checkpoint)


lstm_multi.eval()
with open('../opti/output/lstm_single/best_params_lstm_single_' + forecast_var + '.yaml') as file:
    best_params_uni = yaml.load(file, Loader=yaml.FullLoader)
    optimizer="kaiming"
lstm_uni  = TemperatureModel(hidden_size=best_params_uni["hidden_size"], learning_rate=best_params_uni["learning_rate"], weight_decay=best_params_uni["weight_decay"],optimizer=optimizer,num_layers=best_params_uni["num_layers"],dropout=0,weight_intiliazier=best_params_uni["weight_intiliazier"])
checkpoint_lstm = torch.load('../opti/output/lstm_single/models/best_model_state_' +forecast_var+ '.pt')
lstm_uni.load_state_dict(checkpoint_lstm)  # ['state_dict'])
lstm_uni.eval()
series_air = AirPassengersDataset().load()

models = [NaiveDrift(), NaiveSeasonal(12)]

ensemble_model = RegressionEnsembleModel(
    forecasting_models=models, regression_train_n_points=12
)

#backtest = ensemble_model.historical_forecasts(
 #   series_air, start=0.6, forecast_horizon=3, verbose=True
#)

futures = ensemble_model.predict()
#print("MAPE = %.2f" % (mape(backtest, series_air)))
series_air.plot()
#backtest.plot()
plt.show()