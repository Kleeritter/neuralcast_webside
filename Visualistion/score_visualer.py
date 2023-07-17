import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime
import seaborn as sns
import numpy as np
import math
from sklearn.metrics import mean_squared_error


from Model.funcs.trad import p_ro,sarima
from Model.funcs.visualer_funcs import skill_score
# Passe die folgenden Variablen entsprechend an
forecast_var="temp" #Which variable to forecast
window_size=24*7*4 #How big is the window for training
forecast_horizon=24 #How long to cast in the future
forecast_year=2022 #Which year to forecast
dt = datetime.datetime(forecast_year,1,1,0,0) #+ datetime.timedelta(hours=window_size)



nc_path = '../Data/stunden/'+str(forecast_year)+'_resample_stunden.nc' # Replace with the actual path to your NetCDF file

#references=np.load("sarima/reference_temp_.npy").flatten()
references_path="sarima/dart/temp/672_24.nc"#"forecast_sarima.nc"
lstm_uni_path="forecast_lstm_uni.nc"
lstm_multi_path="forecast_lstm_multi.nc"
tft_path="tft_dart.nc"

nhits="nhits.nc"
lstm_uni=xr.open_dataset(lstm_uni_path).to_dataframe()
lstm_multi=xr.open_dataset(lstm_multi_path).to_dataframe()
tft=xr.open_dataset(tft_path).to_dataframe()
data = xr.open_dataset(nc_path).to_dataframe()
references= xr.open_dataset(references_path).to_dataframe()#[:-24]
nhits=xr.open_dataset(nhits).to_dataframe()


print(len(references),len(data),len(lstm_uni),len(lstm_multi),len(nhits))

def skills_calc(model):
    skills = []
    for start,end in zip(range(0,len(lstm_uni)-forecast_horizon+1,24),range(forecast_horizon,len(lstm_uni)+1,24)):
        actual_values=data[forecast_var][start:end]
        reference_values=references[forecast_var][start:end]
        prediction=model[forecast_var][start:end]
        skill=skill_score(actual_values=actual_values,reference_values=reference_values,prediction=prediction)
        skills.append(skill)
        if skill<-100:
            print("alarm")
            visuals = pd.DataFrame({
                'Datum': data[start:end].index.tolist(),
            #     'Datum': np.arange(0,363,1),
                'Messdaten': actual_values,
                'Modell': prediction,
                'SARIMA': reference_values,
                'Univariantes-LSTM': lstm_uni[forecast_var][start:end],
                'TFT': tft[forecast_var][start:end]
            })
            sns.lineplot(x="Datum", y='value', hue='variable', data=pd.melt(visuals, ['Datum']))
            plt.show()
         #print("reference: ",reference_values)
          #  print("real: ", actual_values)
           # print("forecast: ", prediction)
    print("skills: ", len(np.array(skills)))
    print("mittlerer Skillwert: ", np.median(np.array(skills)), " Abweichung: ", np.std(np.array(skills)))
    return skills





#np.save(file="reference.csv",arr=np.array(referencor))

sns.set_theme(style="darkgrid")


#print(len(data))
#skiller=pd.DataFrame({
    #'Datum': data.index.tolist()[:-24],
    #'Datum': np.arange(0,365,1),
    #'Messdaten': data[forecast_var].tolist(),
    #'Univariantes-LSTM' :lstm_uni[forecast_var],
    #'Multivariantes-LSTM' :lstm_multi[forecast_var],#[:-25],
    #'TFT' :np.array(predicted_temp_tft).flatten(),
    #'SKILL_TFT': np.array(skills_calc(tft)).flatten(),
    #'SKILL_LSTM_uni': np.array(skills_calc(lstm_uni)).flatten(),
    #'SKILL_LSTM_multi': np.array(skills_calc(lstm_multi)).flatten(),
    #'SKILL_nHits': np.array(skills_calc(nhits)).flatten(),
    #'RMSE_TFT':rmse()
    #'SARIMA' : references[:-24]

#})
print(math.sqrt(mean_squared_error(data["temp"], nhits["temp"])))
temps=[]
humids=[]
globals=[]
rain=[]
tagesmax_temps=[]
tagesmin_temps=[]
press=[]
press_3h=[]
diffus=[]
gust_10=[]
gust_50=[]
wind_10=[]
wind_50=[]
wind_dir_50_sin=[]
wind_dir_50_cos=[]
modellist=[references,lstm_uni,lstm_multi,nhits]
def tagesmax(data):
    tagesmax=data.groupby(data.index.date)['temp'].max()
    return tagesmax

def tagesmin(data):
    tagesmin=data.groupby(data.index.date)['temp'].min()
    return tagesmin

def luftdrucktendenz(data):
    luftdrucktendenz = data['press_sl'].resample('3H').mean().diff().fillna(0)/100
    return luftdrucktendenz




for model in modellist:
    #print(model)
    temps.append( math.sqrt(mean_squared_error(data["temp"], model["temp"])))
    tagesmax_temps.append( math.sqrt(mean_squared_error(tagesmax(data),tagesmax(model))))
    tagesmin_temps.append(math.sqrt(mean_squared_error(tagesmin(data), tagesmin(model))))
    humids.append(math.sqrt(mean_squared_error(data["humid"], model["humid"])))
    press.append(math.sqrt(mean_squared_error(data["press_sl"]/100, model["press_sl"]/100)))
    press_3h.append(math.sqrt(mean_squared_error(luftdrucktendenz(data), luftdrucktendenz(model))))
    globals.append(math.sqrt(mean_squared_error(data["globalrcmp11"], model["globalrcmp11"])))
    rain.append(math.sqrt(mean_squared_error(data["rain"], model["rain"])))
    diffus.append(math.sqrt(mean_squared_error(data["diffuscmp11"], model["diffuscmp11"])))
    gust_10.append(math.sqrt(mean_squared_error(data["gust_10"], model["gust_10"])))
    gust_50.append(math.sqrt(mean_squared_error(data["gust_50"], model["gust_50"])))
    wind_10.append(math.sqrt(mean_squared_error(data["wind_10"], model["wind_10"])))
    wind_50.append(math.sqrt(mean_squared_error(data["wind_50"], model["wind_50"])))
    wind_dir_50_sin.append(math.sqrt(mean_squared_error(data["wind_dir_50_sin"], model["wind_dir_50_sin"])))
    wind_dir_50_cos.append(math.sqrt(mean_squared_error(data["wind_dir_50_cos"], model["wind_dir_50_cos"])))





scores=pd.DataFrame({
    'Models': ["SARIMA","LSTN_UNI","LSTM_MULTI","nhits"],
    'RMSE_temp':temps,
    'RMSE_tmax': tagesmax_temps,
    'RMSE_tmin': tagesmin_temps,
    'RMSE_humid':humids,
    "RMSE_rain":rain,
    "RMSE_press":press,
    "RMSE_press_3h":press_3h,
    'RMSE_globals': globals,
    "RMSE_diffus":diffus,
    "RMSE_gust_10":gust_10,
    "RMSE_gust_50":gust_50,
    "RMSE_wind_10":wind_10,
    "RMSE_wind_50":wind_50,
    "RMSE_wind_dir_50":wind_dir_50_sin,
    #"RMSE_wind_dir_50_cos":wind_dir_50_cos

    #'RMSE_TFT':rmse()
    #'SARIMA' : references[:-24]

})
skills=pd.DataFrame({
    'Models': ["LSTN_UNI","LSTM_MULTI","nhits"],
    'temp': [(1-(x/temps[0])) for x in temps[1:]],
    'tmax': [(1-(x/tagesmax_temps[0])) for x in tagesmax_temps[1:]],
    'tmin': [(1-(x/tagesmin_temps[0])) for x in tagesmin_temps[1:]],
    'humid': [(1-(x/humids[0])) for x in humids[1:]],
    'rain': [(1-(x/rain[0])) for x in rain[1:]],
    'press': [(1-(x/press[0])) for x in press[1:]],
    'press_3h': [(1-(x/press_3h[0])) for x in press_3h[1:]],
    'globals': [(1 - (x / globals[0])) for x in globals[1:]],
    'diffus': [(1-(x/diffus[0])) for x in diffus[1:]],
    'gust_10': [(1-(x/gust_10[0])) for x in gust_10[1:]],
    'gust_50': [(1-(x/gust_50[0])) for x in gust_50[1:]],
    'wind_10': [(1-(x/wind_10[0])) for x in wind_10[1:]],
    'wind_50': [(1-(x/wind_50[0])) for x in wind_50[1:]],
    'wind_dir_50': [(1-(x/wind_dir_50_sin[0])) for x in wind_dir_50_sin[1:]],
    'wind_dir_50_cos': [(1-(x/wind_dir_50_cos[0])) for x in wind_dir_50_cos[1:]]

})
#scores=scores.set_index("Models")
skills=skills.set_index("Models")
# Plot the responses for different events and regions
#sns.lineplot(x="Datum", y="SKILL",             data=skiller)

#sns.lineplot(x="Datum", y='value', hue='variable',data=pd.melt(visualerdata, ['Datum']))
#sns.lineplot(x="Datum", y='value', hue='variable',data=pd.melt(skiller, ['Datum']))

#plt.show()


def style_negative(v, props=''):
    return props if v < 0 else None
def highlight_max(s, props=''):
    return np.where(s == np.nanmax(s.values), props, '')
#scores.style \
 # .format(precision=3, thousands=".", decimal=",") \
  #.format_index(str.upper, axis=1) \
  #.applymap(style_negative, props='color:red;')

#scores.apply(highlight_max, props='color:white;background-color:darkblue', axis=0)
#plt.show()
#skills
#skills.style.format(precision=2)\
  #  .highlight_max().to_excel("skillss.xlsx")

#scores.style.format(precision=2)\
 #   .highlight_min().to_excel("scores.xlsx")
  #          #.background_gradient(cmap="RdYlGn").
print(skills)