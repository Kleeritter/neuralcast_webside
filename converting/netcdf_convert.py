import xarray as xr
import pandas as pd


def convert_years(path="/data/datenarchiv/imuk/", year="2022"):

    return

def convert_months(path="/data/datenarchiv/imuk/", year="2022", month="1"):
    
    return

def convert_days(path="/data/datenarchiv/imuk/", year="2022", month="1", day="11"):
    herrenhausen_data= pd.read_csv(path+"hh"+year+month.zfill(2)+day.zfill(2)+".csv")
    print(herrenhausen_data.head())
    return


convert_days()