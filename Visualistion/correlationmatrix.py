import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import scipy.stats as stats
import seaborn as sns
import yaml
nc_path = '../Data/zusammengefasste_datei_2016-2022.nc' # Replace with the actual path to your NetCDF file
data = xr.open_dataset(nc_path).to_dataframe()[[ "temp","press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50",     "rain", "wind_10", "wind_50","wind_dir_50_sin", "wind_dir_50_cos","taupunkt"]]
def resammple(data,var,sum=False):
    if sum:
        sample = data[var].rolling('3H').sum().fillna(0)
    else:
        sample = data[var].rolling('3H').mean().diff().fillna(0)
    return sample

forecast_varsx=[ "temp","press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50",     "rain", "wind_10", "wind_50","wind_dir_50_sin", "wind_dir_50_cos","Taupunkt","Taupunkt3h","press3h","rainsum3h","temp3h","diff wind","rain_event"]
forecast_varsy=[ "temp","press_sl","humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50",     "rain", "wind_10", "wind_50","wind_dir_50_sin", "wind_dir_50_cos" ,"Taupunkt","Taupunkt3h","press3h","rainsum3h","temp3h","diff wind","rain_event"]
data["Taupunkt3h"]=resammple(data,"taupunkt")
data["press3h"]=resammple(data,"press_sl")
data["rainsum3h"]=resammple(data,"rain",sum=True)
data["temp3h"]=resammple(data,"temp")
data["gradwind"] = data["wind_50"]-data["wind_10"]
data["rain_event"]=data["rain"].rolling('3H').apply(lambda x: 1 if x.sum()>0 else 0).fillna(0)

cors=data.corr()


fig, ax = plt.subplots()
# Transformiere die p-Werte
p_value_matrix = data.corr(method=lambda x, y: stats.pearsonr(x, y)[1])
transformed_p_values = -np.log10(p_value_matrix)

# Kombiniere Korrelationskoeffizient und p-Wert
combined_metric = np.abs(cors) * transformed_p_values

cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(cors, dtype=bool))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(cors, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(forecast_varsx)), labels=forecast_varsx,rotation=90)
ax.set_yticks(np.arange(len(forecast_varsy)), labels=forecast_varsy,rotation=0)
ax.set_title("Korrelationsmatrix")


print(p_value_matrix["press_sl"])
def cornames(var):
    corslist = [column for column, correlation in cors[var].items() if abs(correlation) > 0.2 and p_value_matrix[var][column] < 0.05]
    corslist.append(var)
    yaml_datei_pfad = "../Model/corpars/"+var+".yaml"

    # Speichern der Parameter in der YAML-Datei
    with open(yaml_datei_pfad, 'w') as file:
        yaml.dump(corslist, file)
    return
#for var in [ "temp","press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50",     "rain", "wind_10", "wind_50","wind_dir_50_sin", "wind_dir_50_cos","taupunkt"]:
   # cornames(var)
plt.savefig("/home/alex/ranze",dpi=300)