from darts.models import RegressionEnsembleModel, NaiveDrift, NaiveSeasonal,RNNModel
from darts.datasets import AirPassengersDataset
from matplotlib import pyplot as plt
series_air = AirPassengersDataset().load()

models = [NaiveDrift(), NaiveSeasonal(12)]

ensemble_model = RegressionEnsembleModel(
    forecasting_models=models, regression_train_n_points=12
)

backtest = ensemble_model.historical_forecasts(
    series_air, start=0.6, forecast_horizon=3, verbose=True
)

#print("MAPE = %.2f" % (mape(backtest, series_air)))
series_air.plot()
backtest.plot()
plt.show()