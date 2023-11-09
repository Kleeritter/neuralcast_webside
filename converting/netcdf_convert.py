import xarray as xr
import pandas as pd
import yaml
import calendar as cal
#from tqdm import tqdm

def attribute_transfer(xarray_dataset, location="Herrenhausen"):
        # Pfad zur YAML-Datei
    if location== "Herrenhausen":
        yaml_file_path = 'Attributes/attributes_herrenhausen.yml'
    else:
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


def herrenhausen_tools(herrenhausen_data):
    herrenhausen_data.rename(columns={herrenhausen_data.columns[0]: "time",herrenhausen_data.columns[7]: "Wind_Speed",herrenhausen_data.columns[9]: "Gust_Speed"}, inplace=True)
    herrenhausen_data["time"] = pd.to_datetime(herrenhausen_data["time"], format="%d.%m.%Y %H:%M:%S")#'%Y-%m-%d %H:%M:%S')
    herrenhausen_data.set_index('time', inplace=True)
    new_column_names = {col: f'herrenhausen_{col.lstrip()}' for col in herrenhausen_data.columns}
    herrenhausen_data.rename(columns=new_column_names, inplace=True)
    herrenhausen_data=herrenhausen_data[~herrenhausen_data.index.duplicated(keep='last')]
    return herrenhausen_data

def dach_tools(dach_data):
    dach_data.rename(columns={dach_data.columns[0]: "time"}, inplace=True)
    dach_data["time"] = pd.to_datetime(dach_data["time"], format="%d.%m.%Y %H:%M:%S")#'%Y-%m-%d %H:%M:%S')
    dach_data.set_index('time', inplace=True)
    new_column_names = {col: f'dach_{col.lstrip()}' for col in dach_data.columns}
    dach_data.rename(columns=new_column_names, inplace=True)
    dach_data=dach_data[~dach_data.index.duplicated(keep='last')]
    return dach_data


def sonic_tools(sonic_data):
    sonic_data.rename(columns={sonic_data.columns[0]: "time",sonic_data.columns[2]: "Wind_Speed",sonic_data.columns[3]: "Wind_Dir",sonic_data.columns[4]: "Gust_Speed"}, inplace=True)
    sonic_data["time"] = pd.to_datetime(sonic_data["time"], format="%d.%m.%Y %H:%M")#'%Y-%m-%d %H:%M:%S')
    sonic_data.set_index('time', inplace=True)
    new_column_names = {col: f'sonic_{col.lstrip()}' for col in sonic_data.columns}
    sonic_data.rename(columns=new_column_names, inplace=True)
    sonic_data=sonic_data[~sonic_data.index.duplicated(keep='last')]
    return sonic_data


def ruthe_tools(ruthe_data):
    ruthe_data.rename(columns={ruthe_data.columns[0]: "time"}, inplace=True)
    ruthe_data["time"] = pd.to_datetime(ruthe_data["time"], format="%d.%m.%Y %H:%M:%S")#'%Y-%m-%d %H:%M:%S')
    ruthe_data.set_index('time', inplace=True)
    new_column_names = {col: f'ruhte_{col.lstrip()}' for col in ruthe_data.columns}
    ruthe_data.rename(columns=new_column_names, inplace=True)

    return ruthe_data

def mast_tools(mast_data):
    mast_data.rename(columns={mast_data.columns[0]: "time",mast_data.columns[1]: "CO2_15",mast_data.columns[2]: "CO2_10",mast_data.columns[3]: "CO2_2"}, inplace=True)
    mast_data["time"] = pd.to_datetime(mast_data["time"], format="%d.%m.%Y %H:%M:%S")#'%Y-%m-%d %H:%M:%S')
    mast_data.set_index('time', inplace=True)

    new_column_names = {col: f'mast_{col.lstrip()}' for col in mast_data.columns}
    mast_data.rename(columns=new_column_names, inplace=True)
    return mast_data
def convert_years(path="/data/datenarchiv/imuk/", year=2022, month=1,full=True, startday="2022-01-01",endday="2022-03-01", location="Herrenhausen", filename="test_year.nc"):
    if full:
        start_date = str(year)+"-01-01"
        end_date = str(year)+"-12-31"

    else:
        start_date = startday
        end_date = endday
    date_range = pd.date_range(start=start_date, end=end_date)
    daydata = pd.DataFrame()
    for day in date_range:# tqdm( date_range):
        oldday=daydata
        try:
            daydata=convert_singleday(path=path,year=day.strftime('%Y'), month=day.strftime('%m'), day=day.strftime('%d'),location=location)
        except Exception as er:
            print("Day ",day, " not available", er)
            pass

        daydata = pd.concat([oldday, daydata])
    merged_data=daydata.to_xarray()
    merged_data.fillna(-9999)
    merged_data = attribute_transfer(merged_data, location=location)
    merged_data.to_netcdf(filename)
    return merged_data

def convert_months(path="/data/datenarchiv/imuk/", year=2022, month=1,full=True, startday="",endday="", location="Herrenhausen",export=True, filename="test_month.nc"):
    if full:
        num_days = cal.monthrange(year, month)[1]
        start=1
        end= num_days+1
    else:
        start=startday
        end=endday

    daydata = pd.DataFrame()
    for day in range(start,end):
        oldday=daydata
        try:
            daydata=convert_singleday(year=str(year), month=str(month), day=str(day),location=location)
        except:
            print("Day ",day, " not available")
            pass

        daydata = pd.concat([oldday, daydata])
    if export:
        merged_data=daydata.to_xarray()
        merged_data = attribute_transfer(merged_data, location=location)
        merged_data.to_netcdf(filename)
    else:
        merged_data=daydata
    return merged_data
def convert_day(path="/data/datenarchiv/imuk/", year="2022", month="1", day="11", location="Herrenhausen", filename="test_day.nc"):
    
    daydata= convert_singleday(path=path,year=year,month=month,day=day,location=location)
    merged_data=daydata.to_xarray()
    merged_data = attribute_transfer(merged_data, location=location)
    merged_data.to_netcdf(filename)
    return merged_data
def convert_singleday(path="/data/datenarchiv/imuk/", year="2022", month="1", day="11", location="Herrenhausen"):
   
    if location== "Herrenhausen":
        try:
            herrenhausen_data= pd.read_csv(path+"herrenhausen/"+year+"/hh"+year+month.zfill(2)+day.zfill(2)+".csv", delimiter=";")
            herrenhausen_data=herrenhausen_tools(herrenhausen_data)
        except:
            print("Herrenhausen Problem")
        try:
            dach_data = pd.read_csv(path+"dach/"+year+"/kt"+year+month.zfill(2)+day.zfill(2)+".csv", delimiter=";")
            dach_data = dach_tools(dach_data)
        except:
            print("Dach Problem")

        try:
            sonic_data = pd.read_csv(path+"sonic/"+year+"/sonic"+year+month.zfill(2)+day.zfill(2)+".txt", delimiter=";")
            sonic_data =sonic_tools(sonic_data)
        except:
            print("Sonic Problem")
        try:
            merged_data = pd.concat([herrenhausen_data, dach_data, sonic_data], axis=1)

        except:
            print("merge Problem")
            try:
                merged_data = pd.concat([herrenhausen_data, dach_data], axis=1)
            except:
                merged_data =herrenhausen_data
                pass
            #print(herrenhausen_data.head())
            #print(dach_data.head())
            #print(sonic_data.head())
            print(len(herrenhausen_data.index),len(dach_data.index),len(sonic_data.index))
            pass

        merged_data.columns = merged_data.columns.str.replace(r'\s*\(.*\)', '', regex=True)
        merged_data.columns = merged_data.columns.str.replace(' ', '_')
        #merged_data=merged_data.to_xarray()#.to_netcdf("test.nc")
        #merged_data = attribute_transfer(merged_data)
        #merged_data.to_netcdf("test.nc")

    else:
        ruthe_data= pd.read_csv(path+"ruthe/"+year+"/rt"+year+month.zfill(2)+day.zfill(2)+".csv", delimiter=";", encoding="latin-1")
        ruthe_data =ruthe_tools(ruthe_data)
  
        try:

            mast_data= pd.read_csv(path+"ruthemast/"+year+"/rm"+year+month.zfill(2)+day.zfill(2)+".csv", delimiter=";", encoding="latin-1")
            mast_data=mast_tools(mast_data)


            merged_data = pd.concat([ruthe_data, mast_data], axis=1)
        except:
            print("Mast not available")
            merged_data=ruthe_data
            pass
        merged_data.columns = merged_data.columns.str.replace(r'\s*\(.*\)', '', regex=True)
        merged_data.columns = merged_data.columns.str.replace(' ', '_')

        #merged_data=merged_data.to_xarray()#.to_netcdf("test.nc")
        #merged_data = attribute_transfer(merged_data, location="Ruthe")
        #merged_data.to_netcdf("test_ruthe.nc")

    return merged_data


#convert_days()#location="ruthe")

#merged_data=convert_months()

convert_years(year=2022, full=False)
#convert_day(year="2022",month="10",day="11")


