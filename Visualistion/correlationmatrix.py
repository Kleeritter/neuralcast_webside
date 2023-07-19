import numpy as np
import matplotlib
import matplotlib as mpl
#from score_visualer import tagesmax,tagesmin,luftdrucktendenz
import matplotlib.pyplot as plt
from matplotlib import colors
import xarray as xr
import scipy.stats as stats
import seaborn as sns

import math
from sklearn.metrics import mean_squared_error
import yaml
nc_path = '../Data/zusammengefasste_datei_2016-2022.nc' # Replace with the actual path to your NetCDF file
data = xr.open_dataset(nc_path).to_dataframe()[[ "temp","press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50",     "rain", "wind_10", "wind_50","wind_dir_50_sin", "wind_dir_50_cos","taupunkt"]]
def resammple(data,var,sum=False):
    if sum:
        sample = data[var].rolling('3H').sum().fillna(0)
    else:
        sample = data[var].rolling('3H').mean().diff().fillna(0)
    return sample

forecast_varsx=[ "temp","press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50",     "rain", "wind_10", "wind_50","wind_dir_50_sin", "wind_dir_50_cos","Taupunkt","Taupunkt3h","press3h","rainsum3h","temp3h","gradwind","rain_event"]
forecast_varsy=[ "temp","press_sl","humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50",     "rain", "wind_10", "wind_50","wind_dir_50_sin", "wind_dir_50_cos" ,"Taupunkt","Taupunkt3h","press3h","rainsum3h","temp3h","gradwind","rain_event"]
#cors = np.zeros((len(forecast_varsy), len(forecast_varsx)))
data["Taupunkt3h"]=resammple(data,"taupunkt")
data["press3h"]=resammple(data,"press_sl")
data["rainsum3h"]=resammple(data,"rain",sum=True)
data["temp3h"]=resammple(data,"temp")
data["gradwind"] = data["wind_50"]-data["wind_10"]
data["rain_event"]=data["rain"].rolling('3H').apply(lambda x: 1 if x.sum()>0 else 0).fillna(0)

#for i in range(0,len(forecast_varsx)):
 #   for j in range(0,len(forecast_varsy)):
  #      cors[i,j]= data[forecast_varsx[i]].corr()[forecast_varsy[j]]
cors=data.corr()
#corslist= [x for  x in sorted((abs(cors["temp"]))) if x>0.1]

    #print(abs(cors["humid"]))
    #corsnames=[cors["temp"][x].index for x in corslist]
    #print(corslist)
#print(cors["temp"].index[cors["temp"] == corslist[0]])
print(abs(cors["temp"]))
#print( cors["temp"].sort_values(ascending=False)[:5].index.tolist())

fig, ax = plt.subplots()
# Transformiere die p-Werte
p_value_matrix = data.corr(method=lambda x, y: stats.pearsonr(x, y)[1])
transformed_p_values = -np.log10(p_value_matrix)

# Kombiniere Korrelationskoeffizient und p-Wert
combined_metric = np.abs(cors) * transformed_p_values


#im = ax.imshow(cors,cmap="seismic")
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(cors, dtype=bool))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(cors, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(forecast_varsx)), labels=forecast_varsx,rotation=45)
ax.set_yticks(np.arange(len(forecast_varsy)), labels=forecast_varsy,rotation=45)
ax.set_title("Korrelationsmatrix")
# Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#         rotation_mode="anchor")
#cbar = ax.figure.colorbar(im, ax=ax, )
#cbar.ax.set_ylabel("Pearson Korrelationskoeffizient", rotation=-90, va="bottom")

print(p_value_matrix["press_sl"])
def cornames(var):
    corslist = [column for column, correlation in cors[var].items() if abs(correlation) > 0.2 and p_value_matrix[var][column] < 0.05]
    corslist.append(var)
    yaml_datei_pfad = "../Model/corpars/"+var+".yaml"

    # Speichern der Parameter in der YAML-Datei
    with open(yaml_datei_pfad, 'w') as file:
        yaml.dump(corslist, file)
    return
for var in [ "temp","press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50",     "rain", "wind_10", "wind_50","wind_dir_50_sin", "wind_dir_50_cos","taupunkt"]:
    cornames(var)
plt.show()