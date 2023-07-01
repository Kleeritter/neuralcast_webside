from netCDF4 import Dataset
import os
import xarray as xr
import math
import pandas as pd
import yaml
# Erstellen Sie eine neue NetCDF-Datei zum Zusammenfassen der Daten

# Erstellen Sie eine leere Dataset zum Zusammenführen der Daten
combined_dataset = xr.Dataset()

filelist=['zusammengefasste_datei_2016-2019','stunden/2022_resample_stunden.nc']
# Iterieren Sie über die Jahre 2010-2014
for file in filelist:
    input_file = file # Annahme: Die Dateien haben das Format "datei_<jahr>.nc"

    # Überprüfen Sie, ob die Eingabedatei existiert
    if os.path.isfile(input_file):
        input_dataset = xr.open_dataset(input_file)  # Öffnen Sie die Eingabedatei

        # Führen Sie die Eingabedatei mit der kombinierten Dataset zusammen
        combined_dataset = xr.merge([combined_dataset, input_dataset])

        input_dataset.close()  # Schließen Sie die Eingabedatei

datalast= xr.open_dataset('zusammengefasste_datei_2016-2019.nc')
data= xr.open_dataset('stunden/2022_resample_stunden.nc')
data=xr.concat([datalast,data],dim="index").to_dataframe()
# Speichern Sie das kombinierte Dataset in eine neue Datei

#data=combined_dataset.to_dataframe()
print(data.head())
#print("MaxT", max(data['temp']),data.loc[data['temp'] == max(data['temp'])])
#print("MaxH", max(data['humid']),data.loc[data['humid'] == max(data['humid'])])
#print("MaxP", max(data['press_sl']),data.loc[data['press_sl'] == max(data['press_sl'])])
#print("MaxW", max(data['wind_50']),data.loc[data['wind_50'] == max(data['wind_50'])])
#print("MaxW", max(data['wind_10']),data.loc[data['wind_10'] == max(data['wind_10'])])
#print("MaxW", max(data['gust_10']),data.loc[data['gust_10'] == max(data['gust_10'])])
#print("MaxW", max(data['gust_50']),data.loc[data['gust_50'] == max(data['gust_50'])])
#print("MaxR", max(data['rain']),data.loc[data['rain'] == max(data['rain'])])
#print("MaxD", max(data['diffuscmp11']),data.loc[data['diffuscmp11'] == max(data['diffuscmp11'])])
#print("MaxG", max(data['globalrcmp11']),data.loc[data['globalrcmp11'] == max(data['globalrcmp11'])])
data["wind_dir_sin"]=data["wind_dir_50"].apply(lambda x: math.sin(math.radians(x)))
data["wind_dir_cos"]=data["wind_dir_50"].apply(lambda x: math.cos(math.radians(x)))
#print("MaxW", max(data['wind_dir_sin']),data.loc[data['wind_dir_sin'] == max(data['wind_dir_sin'])])
#print("MaxW", max(data['wind_dir_cos']),data.loc[data['wind_dir_cos'] == max(data['wind_dir_cos'])])

maxT=round(max(data['temp']),2)
maxH=round(max(data['humid']),2)
maxP=round(max(data['press_sl']),2)
maxW=round(max(data['wind_50']),2)
maxW10=round(max(data['wind_10']),2)
maxG=round(max(data['gust_50']),2)
maxG10=round(max(data['gust_10']),2)
maxR=round(max(data['rain']),2)
maxD=round(max(data['diffuscmp11']),2)
maxGlo=round(max(data['globalrcmp11']),2)
maxSin=round(max(data['wind_dir_sin']),2)
maxCos=round(max(data['wind_dir_cos']),2)

minT=round(min(data['temp']),2)
minH=round(min(data['humid']),2)
minP=round(min(data['press_sl']),2)
minW=round(min(data['wind_50']),2)
minW10=round(min(data['wind_10']),2)
minG=round(min(data['gust_50']),2)
minG10=round(min(data['gust_10']),2)
minR=round(min(data['rain']),2)
minD=round(min(data['diffuscmp11']),2)
minGlo=round(min(data['globalrcmp11']),2)
minSin=round(min(data['wind_dir_sin']),2)
minCos=round(min(data['wind_dir_cos']),2)

yaml_dict = {"Max_temp": maxT, "Max_humid": maxH, "Max_press_sl": maxP, "Max_wind_50": maxW, "Max_wind_10": maxW10, "Max_gust_50": maxG, "Max_gust_10": maxG10, "Max_rain": maxR, "Max_diffuscmp11": maxD, "Max_globalrcmp11": maxGlo, "Max_wind_dir_50_sin": maxSin, "Max_wind_dir_cos": maxCos, "Min_temp": minT, "Min_humid": minH, "Min_press_sl": minP, "Min_wind_50": minW, "Min_wind_10": minW10, "Min_gust_50": minG, "Min_gust_10": minG10, "Min_rain": minR, "Min_diffuscmp11": minD, "Min_globalrcmp11": minGlo, "Min_wind_dir_50_sin": minSin, "Min_wind_dir_50_cos": minCos}

# Schließen Sie das kombinierte Dataset

with open('params_for_normal.yaml', 'w') as file:
    yaml.dump(yaml_dict, file)
combined_dataset.close()
