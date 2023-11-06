import xarray as xr
import pandas as pd


def convert_years(path="/data/datenarchiv/imuk/", year="2022"):

    return

def convert_months(path="/data/datenarchiv/imuk/", year="2022", month="1"):
    
    return

def convert_days(path="/data/datenarchiv/imuk/", year="2022", month="1", day="11", location="Herrenhausen"):
   
    if location== "Herrenhausen":
        herrenhausen_data= pd.read_csv(path+"herrenhausen/"+year+"/hh"+year+month.zfill(2)+day.zfill(2)+".csv", delimiter=";")
        print(herrenhausen_data.head())

        dach_data = pd.read_csv(path+"dach/"+year+"/kt"+year+month.zfill(2)+day.zfill(2)+".csv", delimiter=";")
        print(dach_data.head())

        sonic_data = pd.read_csv(path+"sonic/"+year+"/sonic"+year+month.zfill(2)+day.zfill(2)+".txt", delimiter=";")
        print(sonic_data.head())
    else:
        ruthe_data= "ruthe"


    return


convert_days()