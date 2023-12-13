import xarray as xr
import pandas as pd
import yaml
import calendar as cal
#from tqdm import tqdm

def attribute_transfer(xarray_dataset, location="Herrenhausen"):
   # import os
   # os.chdir("/Users/alex/Code/neuralcast_webside/converting")
        # Pfad zur YAML-Datei
    import os
    origin =os.getcwd()
    print(origin)
    os.chdir(origin +"/neuralcast_webside/converting/")
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



def herrenhausen_tools(herrenhausen_data, format="%d.%m.%Y %H:%M:%S"):
    herrenhausen_data.rename(columns={herrenhausen_data.columns[0]: "time",herrenhausen_data.columns[7]: "Wind_Speed",herrenhausen_data.columns[9]: "Gust_Speed"}, inplace=True)
    herrenhausen_data["time"] = pd.to_datetime(herrenhausen_data["time"], format=format)#'%Y-%m-%d %H:%M:%S')
    herrenhausen_data.set_index('time', inplace=True)
    new_column_names = {col: f'herrenhausen_{col.lstrip()}' for col in herrenhausen_data.columns}
    herrenhausen_data.rename(columns=new_column_names, inplace=True)
    herrenhausen_data=herrenhausen_data[~herrenhausen_data.index.duplicated(keep='last')]
    return herrenhausen_data

def dach_tools(dach_data,  format="%d.%m.%Y %H:%M:%S"):
    dach_data.rename(columns={dach_data.columns[0]: "time"}, inplace=True)
    dach_data["time"] = pd.to_datetime(dach_data["time"], format=format)#'%Y-%m-%d %H:%M:%S')
    dach_data.set_index('time', inplace=True)
    new_column_names = {col: f'dach_{col.lstrip()}' for col in dach_data.columns}
    dach_data.rename(columns=new_column_names, inplace=True)
    dach_data=dach_data[~dach_data.index.duplicated(keep='last')]
    #print(dach_data.columns)
    if "dach_CMP-11 Diffus (W/m2)" in dach_data.columns:
        dach_data.rename(columns={"dach_CMP-11 Diffus (W/m2)":"dach_Diffus_CMP-11"}, inplace=True)
    #else if

    #else:
     #   pass

    return dach_data


def sonic_tools(sonic_data,  format="%d.%m.%Y %H:%M"):
    sonic_data.rename(columns={sonic_data.columns[0]: "time",sonic_data.columns[2]: "Wind_Speed",sonic_data.columns[3]: "Wind_Dir",sonic_data.columns[4]: "Gust_Speed"}, inplace=True)
    sonic_data["time"] = pd.to_datetime(sonic_data["time"], format=format)#'%Y-%m-%d %H:%M:%S')
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
    minutenspanne = pd.date_range(start=daydata.index.min(), end=daydata.index.max(), freq='1T')
    daydata = pd.DataFrame(index=minutenspanne).join(daydata)
    daydata = daydata[daydata.columns[~(daydata.columns.str.contains('Unnamed') | daydata.columns.str.contains(r'\.\d'))]]
    daydata=daydata[~daydata.index.duplicated(keep='last')]
    daydata.index.names = ['time']


    merged_data=daydata.to_xarray()
    spaltennamen =["dach_CO2_ppm","dach_CO2_Sensor","dach_Diffus_CMP-11","dach_Geneigt_CM-11","dach_Global_CMP-11",
    "dach_Temp_AMUDIS_Box","dach_Temp_Bentham_Box","herrenhausen_Druck","herrenhausen_Feuchte","herrenhausen_Gust_Speed"
    ,"herrenhausen_Psychro_T","herrenhausen_Psychro_Tf","herrenhausen_Pyranometer_CM3","herrenhausen_Regen","herrenhausen_Temperatur","herrenhausen_Wind_Speed",
    "sonic_Gust_Speed","sonic_Temperatur","sonic_Wind_Dir","sonic_Wind_Speed"]
    for spaltenname in spaltennamen:
        if spaltenname  not in merged_data.data_vars:
            merged_data[spaltenname] = xr.DataArray( )  # Hinzufügen der neuen Spalte mit NaN-Werten

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
   
    def read_and_process_data(folder, prefix, year, month, day, file_extension, tools_function, encoding="utf8"):
        try:
            file_path = path + folder + year + "/" + prefix + year + month.zfill(2) + day.zfill(2) + file_extension
            if file_extension == ".csv":
                data = pd.read_csv(file_path, delimiter=";",encoding=encoding)
            elif file_extension == ".txt":
                data = pd.read_csv(file_path, delimiter=";",encoding=encoding)  # Annahme: Tabulator als Trennzeichen in der TXT-Datei
            else:
                raise ValueError("Ungültige Datei-Erweiterung")

            data = tools_function(data)
            data = data.apply(pd.to_numeric, errors='coerce')


            return data
        except Exception as e:
            try: 
                data = tools_function(data, format="%d.%m.%y %H:%M:%S")
            except:
                try: 
                    data = tools_function(data, format="%d.%m.%Y %H:%M")
                except:
                    print(f"{prefix} Problem: {e}")
            return None
    if location== "Herrenhausen":
            # Verwendung der Funktionen
            herrenhausen_data = read_and_process_data("herrenhausen/", "hh", year, month, day, ".csv", herrenhausen_tools)
            dach_data = read_and_process_data("dach/", "kt", year, month, day, ".csv", dach_tools)
            sonic_data = read_and_process_data("sonic/", "sonic", year, month, day, ".txt", sonic_tools)


            # Zusammenführen der Daten
            data_list = [herrenhausen_data, dach_data, sonic_data]
            merged_data = pd.concat([data for data in data_list if data is not None], axis=1)
   
           
            merged_data.columns = merged_data.columns.str.replace(r'\s*\(.*\)', '', regex=True)
            merged_data.columns = merged_data.columns.str.replace(' ', '_')


    else:
        ruthe_data=  read_and_process_data("ruthe/", "rt", year, month, day, ".csv", ruthe_tools, encoding="latin-1")
        mast_data =read_and_process_data("ruthemast/", "rm", year, month, day, ".csv", mast_tools,encoding="latin-1")
        data_list = [ruthe_data, mast_data]
        merged_data = pd.concat([data for data in data_list if data is not None], axis=1)
        """
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
        """
        merged_data.columns = merged_data.columns.str.replace(r'\s*\(.*\)', '', regex=True)
        merged_data.columns = merged_data.columns.str.replace(' ', '_')

        #merged_data=merged_data.to_xarray()#.to_netcdf("test.nc")
        #merged_data = attribute_transfer(merged_data, location="Ruthe")
        #merged_data.to_netcdf("test_ruthe.nc")

    return merged_data


#convert_days()#location="ruthe")

#merged_data=convert_months()

#convert_years(year=2022, full=False)
#convert_day(year="2022",month="10",day="11")


