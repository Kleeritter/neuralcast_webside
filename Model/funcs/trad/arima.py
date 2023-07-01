def arima(data):
    import pandas as pd
    import statsmodels.api as sm


    train= data
    train.index = pd.DatetimeIndex(train.index.values,
                                   freq="H")
    # Erstellen und fitten Sie das SARIMA-Modell
    order = (0, 1, 1)  # Beispielwerte für AR, I, MA
    #seasonal_order = (0, 1, 1, 24)  # Beispielwerte für saisonale AR, I, MA, Saisonlänge
    model = sm.tsa.statespace.SARIMAX(train, order=order)
    model_fit = model.fit(disp=0)

    # Generieren Sie Vorhersagen für die nächsten 24h
    forecast = model_fit.get_forecast(steps=24)

    # Extrahieren Sie die Vorhersageergebnisse
    forecast_values = forecast.predicted_mean
    confidence_interval = forecast.conf_int()

    return forecast_values.values


