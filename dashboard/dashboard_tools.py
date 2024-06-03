from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
# Funktion zur Berechnung des Trends
def calculate_trend(df, hours=3):
    # Zeitfilter für die letzten 3 Stunden
    end_time = df.index.max()
    start_time = end_time - pd.Timedelta(hours=hours)
    
    # Filtere die letzten 3 Stunden
    df_last_hours = df.loc[start_time:end_time]
    
    # Berechne die stündlichen Mittelwerte
    df_hourly = df_last_hours.resample('H').mean()
    
    # Vorbereitung der Daten für die lineare Regression
    X = np.array((df_hourly.index - df_hourly.index[0]).total_seconds()).reshape(-1, 1)
    y = df_hourly['herrenhausen_Druck'].values.reshape(-1, 1)
    
    # Lineare Regression anwenden
    model = LinearRegression().fit(X, y)
    
    # Steigung (Trend) aus dem Modell extrahieren
    trend = model.coef_[0][0]

    trend_per_3h = np.round(trend * 10800,2)  # 10800 Sekunden = 3 Stunden

    
    return trend_per_3h

# Trend der letzten 3 Stunden berechnen
#trend = calculate_trend(df)
#print(f'Trend der letzten 3 Stunden: {trend} hPa pro Sekunde')
