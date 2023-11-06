import xarray as xr
import pandas as pd


def convert_years(path="/data/datenarchiv/imuk/", year="2022"):

    return

def convert_months(path="/data/datenarchiv/imuk/", year="2022", month="1"):
    
    return

def convert_days(path="/data/datenarchiv/imuk/", year="2022", month="1", day="11", location="Herrenhausen"):
   
    if location== "Herrenhausen":
        herrenhausen_data= pd.read_csv(path+"herrenhausen/"+year+"/hh"+year+month.zfill(2)+day.zfill(2)+".csv", delimiter=";")
        herrenhausen_data.rename(columns={herrenhausen_data.columns[0]: "time"}, inplace=True)
        herrenhausen_data["time"] = pd.to_datetime(herrenhausen_data["time"], format="%d.%m.%Y %H:%M:%S")#'%Y-%m-%d %H:%M:%S')
        herrenhausen_data.set_index('time', inplace=True)
        new_column_names = {col: f'herrenhausen_{col.lstrip()}' for col in herrenhausen_data.columns}
        herrenhausen_data.rename(columns=new_column_names, inplace=True)
        print(herrenhausen_data.head())

        dach_data = pd.read_csv(path+"dach/"+year+"/kt"+year+month.zfill(2)+day.zfill(2)+".csv", delimiter=";")
        dach_data.rename(columns={dach_data.columns[0]: "time"}, inplace=True)
        dach_data["time"] = pd.to_datetime(dach_data["time"], format="%d.%m.%Y %H:%M:%S")#'%Y-%m-%d %H:%M:%S')
        dach_data.set_index('time', inplace=True)
        new_column_names = {col: f'dach_{col.lstrip()}' for col in dach_data.columns}
        dach_data.rename(columns=new_column_names, inplace=True)
        print(dach_data.head())

        sonic_data = pd.read_csv(path+"sonic/"+year+"/sonic"+year+month.zfill(2)+day.zfill(2)+".txt", delimiter=";")
        sonic_data.rename(columns={sonic_data.columns[0]: "time"}, inplace=True)
        sonic_data["time"] = pd.to_datetime(sonic_data["time"], format="%d.%m.%Y %H:%M")#'%Y-%m-%d %H:%M:%S')
        sonic_data.set_index('time', inplace=True)
        new_column_names = {col: f'sonic_{col.lstrip()}' for col in sonic_data.columns}
        sonic_data.rename(columns=new_column_names, inplace=True)
        print(sonic_data.head())

        merged_data = pd.concat([herrenhausen_data, dach_data, sonic_data], axis=1)
        merged_data.columns = merged_data.columns.str.replace(r'\s*\(.*\)', '', regex=True)

        print(merged_data.head())

        merged_data=merged_data.to_xarray()#.to_netcdf("test.nc")
        print(merged_data.head())

    else:
        ruthe_data= "ruthe"


    return


convert_days()