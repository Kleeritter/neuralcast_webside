import pandas as pd
import numpy as np
import xarray as xr
import glob
from converterfuncs import dewpointer,dew_pointa, dewpoint_new

towerfolder = "/home/alex/Dokumente/ruthemast/2020/"
folder = "/home/alex/Dokumente/ruthe/2020/"
output = "ruthe_2020.nc"
towerfiles = sorted(glob.glob(towerfolder+"*.csv") )
files = sorted(glob.glob(folder +"*.csv"))

datas=pd.DataFrame()
#print(files[0])
for i in range(len(towerfiles)):
    data = pd.read_csv(towerfiles[i],delimiter=";",encoding="latin-1", dayfirst=True)
    new_col_names={"         Datum/Zeit":"time", "CO2 (15m)": "CO2_15","CO2 (10m)" :"CO2_10","CO2 (2m )":"CO2_2","WindDir (°)": "wind_dir_10", "WindSpeed(m/s)": "wind_10","WindMax(m/s)":"wind_10_max"}
    data.rename(columns=new_col_names,inplace=True)
    data["time"]=pd.to_datetime(data["time"],dayfirst=True)
    data=data.set_index("time")

    data_sup=pd.read_csv(files[i],delimiter=";", encoding="latin-1")
    data_sup = data_sup.rename(columns=lambda x: x.strip())
    new_col_names={"Datum/Zeit": "time","Temp":"temp","Feuchte":"humid","Regen":"rain","Globalstrahlung":"globalrcmp11"}
    data_sup.rename(columns=new_col_names,inplace=True)
    #print(data_sup.columns)
    data_sup["time"]=pd.to_datetime(data_sup["time"],dayfirst=True)
    data_sup=data_sup.set_index("time")

    #print(data_sup.columns)
    data["temp"]=data_sup["temp"] +273.15
    data["humid"]=data_sup["humid"].apply(lambda x: x if x< 1 else x/1000)
    data["rain"]=data_sup["rain"]
    try:
        data["globalrcmp11"]=data_sup["globalrcmp11"]
    except:
        pass
    data["tau"]= data.apply(lambda row: dewpoint_new(row['temp'], row['humid']), axis=1)#dewpointer(data["temp"],data["humid"])
    datas=pd.concat([datas,data])

print(datas.tail())
datas.to_xarray().to_netcdf(output)