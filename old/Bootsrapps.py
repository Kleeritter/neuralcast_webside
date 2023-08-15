import xarray as xr
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm


nc_path = '../Data/stunden/'+str(2022)+'_resample_stunden.nc' # Replace with the actual path to your NetCDF file
baseline_path="../Model/baseline/baseline.nc"
references_path="auto_arima.nc"
lstm_uni_path="forecast_lstm_uni.nc"
lstm_multi_path="time_test_better_a.nc"#"../Model/timetest/lstm_multi/output/temp/timetest_lstm_multitemp_24_24.nc"#"forecast_lstm_multi.nc"
tft_path="tft_dart.nc"
cors_path="cortest_all.nc"#"../Model/cortest/lstm_multi/output/temp/cortest_lstm_multitemp_24_24.nc"
nhits="nhit.nc"
prohet_path="prophet.nc"
lstm_multi_cor="cortest_all_p.nc"

ruthe_path="../Data/ruthe_2020_resample_stunden.nc"
ruthe=xr.open_dataset(ruthe_path).to_dataframe()
ruthe_forecast="ruthe_forecast.nc"
ruthe_forecast=xr.open_dataset(ruthe_forecast).to_dataframe()

baseline=xr.open_dataset(baseline_path).to_dataframe()
lstm_uni=xr.open_dataset(lstm_uni_path).to_dataframe()
lstm_multi=xr.open_dataset(lstm_multi_path).to_dataframe()
LSTM_MULTI_CORS=xr.open_dataset(cors_path).to_dataframe()
tft=xr.open_dataset(tft_path).to_dataframe()
data = xr.open_dataset(nc_path).to_dataframe()
references= xr.open_dataset(references_path).to_dataframe()#[:-24]
nhits=xr.open_dataset(nhits).to_dataframe()
prophet=xr.open_dataset(prohet_path).to_dataframe()
lstm_multi_cor=xr.open_dataset(lstm_multi_cor).to_dataframe()

def checknans(data):
    for var in data.columns:
        print(var, data[var].isna().sum())
    return
def bootstrap_rmse(data1, data2, n_iterations=1000,var="temp"):
    np.random.seed(42)  # Für die Reproduzierbarkeit der Ergebnisse
    n_samples = len(data1)

    rmse_values = []
    for _ in range(n_iterations):
        sample_indices = np.random.randint(0, n_samples, n_samples)
        sample_data1 = data1.iloc[sample_indices]
        sample_data2 = data2.iloc[sample_indices]

        rmse = np.sqrt(mean_squared_error(sample_data1, sample_data2))


        rmse_values.append(rmse)
    print(rmse_values)
    return rmse_values


def bootstrap_mase(data1, data2, n_iterations=1000,var="temp"):
        np.random.seed(42)  # Für die Reproduzierbarkeit der Ergebnisse
        n_samples = len(data1)

        mase_values = []
        for _ in range(n_iterations):
            sample_indices = np.random.randint(0, n_samples, n_samples)
            sample_data1 = data1.iloc[sample_indices]
            basis = baseline[var].iloc[sample_indices]
            measured= data[var].iloc[sample_indices]

            # rmse = np.sqrt(mean_squared_error(sample_data1, sample_data2))
            mase_upper =mean_absolute_error(measured, sample_data1)
            mase_under = mean_absolute_error(measured, basis)
            mase = mase_upper/mase_under

            mase_values.append(mase)

        return mase_values
def boots(model,data=data, var="temp"):

    target_parameter = var  # Ersetzen Sie 'parameter_name' durch den tatsächlichen Namen des Parameters

    # Extrahieren Sie die Zeitreihen für den Zielparameter aus beiden DataFrames
    data1_target = model[target_parameter]

    data2_target = data[target_parameter]

    # Berechnen Sie den RMSE mit dem Bootstrapping-Verfahren
    rmse_values = bootstrap_rmse(data1_target, data2_target, n_iterations=1000,var=var)
    mase_values = bootstrap_mase(data1_target, data2_target, n_iterations=1000,var=var)
    # Zeigen Sie die Ergebnisse an, zum Beispiel den durchschnittlichen RMSE und die Standardabweichung
    average_rmse = np.mean(rmse_values)
    std_dev_rmse = np.std(rmse_values)
    #print(f"Durchschnittlicher RMSE: {average_rmse}")
    #print(f"Standardabweichung des RMSE: {std_dev_rmse}")
    return [rmse_values,mase_values]


#plt.figure(figsize=(8, 6))
#plt.grid(True)

#plt.hist(rmse_values, bins='auto', edgecolor='k')
#plt.xlabel('RMSE')
#plt.ylabel('Häufigkeit')
#plt.title('Histogramm der RMSE-Werte')
def skillboots(data=data):
    varlist=["temp", "humid", "diffuscmp11", "gust_10", "gust_50",     "rain", "wind_10", "wind_50","globalrcmp11","press_sl","wind_dir_50_sin","wind_dir_50_cos"]
    #varlist=["press_sl"]

    modellist=[references,lstm_uni,lstm_multi,lstm_multi_cor,baseline]
    models=["SARIMA","LSTM_UNI","LSTM_MULTI,","LSTM_MULTI_COR","baseline"]
    datas=pd.DataFrame(index=models)
    for var in tqdm(varlist):
        vara=[]
        var_std=[]
        mases = []
        for model in modellist:
             vara.append(np.mean(boots(model,var=var)[0]))
             var_std.append(np.std(boots(model,var=var)[0]))
             mases.append(np.mean(boots(model, var=var)[1]))
        datas[var]=vara
        datas[var+"_std"]=var_std
        datas[var+"_mase"]=mases

    return datas

#plt.show()

#sarima = boots(references)
#lstm_uni = boots(lstm_uni)
#lstm_multi = boots(lstm_multi)
checknans(references)
goose = skillboots()
goose =round(goose,3)
# Spaltennamen sortieren, um _mase-Spalten ans Ende zu verschieben
columns = goose.columns.tolist()
columns.sort(key=lambda x: x.endswith('_mase'))

# DataFrame mit sortierten Spaltennamen erstellen
goose = goose[columns]
#goose=goose.transpose()
print(goose)
goose.to_csv("all672_mases.csv")
#print(np.mean(sarima), np.std(sarima))
#print(np.mean(lstm_uni), np.std(lstm_uni))
#print(np.mean(lstm_multi), np.std(lstm_multi))