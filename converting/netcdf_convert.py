import xarray as xr
import pandas as pd
import yaml

def attribute_transfer(xarray_dataset):
        # Pfad zur YAML-Datei
    yaml_file_path = 'Attributes/attributes_ruthe.yml'

    # YAML-Datei einlesen
    with open(yaml_file_path, 'r') as yaml_file:
        attribute_data = yaml.safe_load(yaml_file)

    # Aktualisieren der Attribute der "temperatur"-Variablen
    vars= xarray_dataset.keys()
    for var in vars:
        if var in attribute_data:
            for key, value in attribute_data[var].items():
                xarray_dataset[var].attrs[key] = value
    return xarray_dataset

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
        merged_data.columns = merged_data.columns.str.replace(' ', '_')

        print(merged_data.head())

        merged_data=merged_data.to_xarray()#.to_netcdf("test.nc")
        print(merged_data.head())
        merged_data = attribute_transfer(merged_data)
        merged_data.to_netcdf("test.nc")

    else:
        ruthe_data= pd.read_csv(path+"ruthe/"+year+"/rt"+year+month.zfill(2)+day.zfill(2)+".csv", delimiter=";", encoding="latin-1")
        ruthe_data.rename(columns={ruthe_data.columns[0]: "time"}, inplace=True)
        ruthe_data["time"] = pd.to_datetime(ruthe_data["time"], format="%d.%m.%Y %H:%M:%S")#'%Y-%m-%d %H:%M:%S')
        ruthe_data.set_index('time', inplace=True)
        new_column_names = {col: f'ruhte_{col.lstrip()}' for col in ruthe_data.columns}
        ruthe_data.rename(columns=new_column_names, inplace=True)
        print(ruthe_data.head())

        try:

            mast_data= pd.read_csv(path+"ruthemast/"+year+"/rm"+year+month.zfill(2)+day.zfill(2)+".csv", delimiter=";", encoding="latin-1")
            mast_data.rename(columns={mast_data.columns[0]: "time",mast_data.columns[1]: "CO2_15",mast_data.columns[2]: "CO2_10",mast_data.columns[3]: "CO2_2"}, inplace=True)
            mast_data["time"] = pd.to_datetime(mast_data["time"], format="%d.%m.%Y %H:%M:%S")#'%Y-%m-%d %H:%M:%S')
            mast_data.set_index('time', inplace=True)
    
            new_column_names = {col: f'mast_{col.lstrip()}' for col in mast_data.columns}
            mast_data.rename(columns=new_column_names, inplace=True)
            print(mast_data.head())

            merged_data = pd.concat([ruthe_data, mast_data], axis=1)
        except:
            print("Mast not available")
            merged_data=ruthe_data
            pass
        merged_data.columns = merged_data.columns.str.replace(r'\s*\(.*\)', '', regex=True)
        merged_data.columns = merged_data.columns.str.replace(' ', '_')

        print(merged_data.head())

        merged_data=merged_data.to_xarray()#.to_netcdf("test.nc")
        print(merged_data.head())
        print(merged_data.keys())
        merged_data = attribute_transfer(merged_data)
        merged_data.to_netcdf("test_ruthe.nc")

    return


convert_days(location="ruthe")


