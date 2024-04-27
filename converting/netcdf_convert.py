#### Skripte die Zum Konvertieren der Daten benötigt werden.

### Das Wichtigste Skript ist hier convert_years, welches für einen Beliebigen Zeitraum konvertiert
### Die convert_singleday Funktion ist Treiber der convert_years Funktion. Hier werden für jeden Tag die Werte eingelesen
### Die einzelnen _tools Funktionen sind wichtig um die Besonderheiten und Unregelmäßigkeiten der verschiednenen Datenquellen zu berücksichtigen
### Die attribute_transfer Funktion erstellt die Attribute der Variablen in der Netcdf Datei, z.B. Longname etc.




import xarray as xr
import pandas as pd
import yaml
import calendar as cal

def attribute_transfer(xarray_dataset, location="Herrenhausen"):
    ### Funktion um die Attribute der Variablen in der NetCDF Datei zu definieren
    import os
    ### Es wird eine YML Datei eingelesen in der für jeden Parameter Attribute definiert sind. Eine für Herrenhausen und eine für Ruthe

    if location== "Herrenhausen":
        yaml_file_path = 'Attributes/attributes_herrenhausen.yml'
    else:
        yaml_file_path = 'Attributes/attributes_ruthe.yml'

    # YAML-Datei einlesen
    with open(yaml_file_path, 'r') as yaml_file:
        attribute_data = yaml.safe_load(yaml_file)

    # Aktualisieren der Attribute der einzelnen Variablen
    vars= xarray_dataset.keys()
    for var in vars:
        if var in attribute_data:
            for key, value in attribute_data[var].items():
                xarray_dataset[var].attrs[key] = value
    return xarray_dataset



def herrenhausen_tools(herrenhausen_data, format="%d.%m.%Y %H:%M:%S"):
    ### Toolfunktion um Probleme in der Herrenhausenquelle zu lösen
    herrenhausen_data.rename(columns={herrenhausen_data.columns[0]: "time",herrenhausen_data.columns[7]: "Wind_Speed",herrenhausen_data.columns[9]: "Gust_Speed"}, inplace=True) #Spalten umbenenen
    herrenhausen_data["time"] = pd.to_datetime(herrenhausen_data["time"], format=format) #Convertieren der Zeit in ein Datetime Objekt
    herrenhausen_data.set_index('time', inplace=True) # Zeit als Index setzen
    new_column_names = {col: f'herrenhausen_{col.lstrip()}' for col in herrenhausen_data.columns} # "herrenhausen_" als Präfix für die Variablen setzen
    herrenhausen_data.rename(columns=new_column_names, inplace=True)
    herrenhausen_data=herrenhausen_data[~herrenhausen_data.index.duplicated(keep='last')] # Duplizierte Werte finden und jeweils den letzen behalten
    return herrenhausen_data

def dach_tools(dach_data,  format="%d.%m.%Y %H:%M:%S"):
    ### Toolfunktion um Probleme in der Dachquelle zu lösen
    dach_data.rename(columns={dach_data.columns[0]: "time"}, inplace=True) # Spalten umbennen
    dach_data["time"] = pd.to_datetime(dach_data["time"], format=format) #Convertieren der Zeit in ein Datetime Objekt
    dach_data.set_index('time', inplace=True) # Zeit als Index setzen
    new_column_names = {col: f'dach_{col.lstrip()}' for col in dach_data.columns} # "dach" als Präfix für die Variablen setzen
    dach_data.rename(columns=new_column_names, inplace=True)
    dach_data=dach_data[~dach_data.index.duplicated(keep='last')] # Duplizierte Werte finden und jeweils den letzen behalten
    if "dach_CMP-11 Diffus (W/m2)" in dach_data.columns: 
        dach_data.rename(columns={"dach_CMP-11 Diffus (W/m2)":"dach_Diffus_CMP-11"}, inplace=True) #Manchmal in neuen bzw. häufig in alten Daten sind die Spaltennamen anders

    return dach_data


def sonic_tools(sonic_data,  format="%d.%m.%Y %H:%M"):
    ### Toolfunktion um Probleme in der sonicquelle zu lösen
    sonic_data.rename(columns={sonic_data.columns[0]: "time",sonic_data.columns[2]: "Wind_Speed",sonic_data.columns[3]: "Wind_Dir",sonic_data.columns[4]: "Gust_Speed"}, inplace=True) # Spalten umbennen
    sonic_data["time"] = pd.to_datetime(sonic_data["time"], format=format) #Convertieren der Zeit in ein Datetime Objekt
    sonic_data.set_index('time', inplace=True) # Zeit als Index setzen
    new_column_names = {col: f'sonic_{col.lstrip()}' for col in sonic_data.columns} # "sonic" als Präfix für die Variablen setzen
    sonic_data.rename(columns=new_column_names, inplace=True)
    sonic_data=sonic_data[~sonic_data.index.duplicated(keep='last')] # Duplizierte Werte finden und jeweils den letzen behalten
    return sonic_data


def ruthe_tools(ruthe_data):
    ### Toolfunktion um Probleme in der Ruthequelle zu lösen
    ruthe_data.rename(columns={ruthe_data.columns[0]: "time"}, inplace=True)  # Spalten umbennen
    ruthe_data["time"] = pd.to_datetime(ruthe_data["time"], format="%d.%m.%Y %H:%M:%S") #Convertieren der Zeit in ein Datetime Objekt
    ruthe_data.set_index('time', inplace=True) # Zeit als Index setzen
    new_column_names = {col: f'ruhte_{col.lstrip()}' for col in ruthe_data.columns}  # "ruthe" als Präfix für die Variablen setzen
    ruthe_data.rename(columns=new_column_names, inplace=True) # Duplizierte Werte finden und jeweils den letzen behalten
    return ruthe_data

def mast_tools(mast_data):
    ### Toolfunktion um Probleme in der Mastquelle zu lösen
    mast_data.rename(columns={mast_data.columns[0]: "time",mast_data.columns[1]: "CO2_15",mast_data.columns[2]: "CO2_10",mast_data.columns[3]: "CO2_2"}, inplace=True) # Spalten umbennen
    mast_data["time"] = pd.to_datetime(mast_data["time"], format="%d.%m.%Y %H:%M:%S") #Convertieren der Zeit in ein Datetime Objekt
    mast_data.set_index('time', inplace=True) # Zeit als Index setzen

    new_column_names = {col: f'mast_{col.lstrip()}' for col in mast_data.columns}  # "mast" als Präfix für die Variablen setzen
    mast_data.rename(columns=new_column_names, inplace=True)
    return mast_data


def convert_years(path="/data/datenarchiv/imuk/", year=2022, month=1,full=True, startday="2022-01-01",endday="2022-03-01", location="Herrenhausen", filename="test_year.nc"):
    ### Konvertieren der Daten eines belibigen Zeitraums
    
    #### Wenn full dann einfach für das ganze aktuelle Jahr
    if full:
        start_date = str(year)+"-01-01"
        end_date = str(year)+"-12-31"
    #### Andernfalls werden Start und Ende aus der Funktion entnommen
    else:
        start_date = startday
        end_date = endday
    
    date_range = pd.date_range(start=start_date, end=end_date) # Range des Datums
    daydata = pd.DataFrame() # Dataframe zum Speichern der eingelesen Daten
    for day in date_range: # Für jeden Tag in der Datumsrange
        oldday=daydata # Umspeichern der Werte
        
        try: # Probieren die Konvertierungsfunktion auszuführen
            daydata=convert_singleday(path=path,year=day.strftime('%Y'), month=day.strftime('%m'), day=day.strftime('%d'),location=location)
        except Exception as er:
            print("Day ",day, " not available", er) #Falls Probleme auftreten Fehler zurückgeben und fortfahren
            pass

        daydata = pd.concat([oldday, daydata]) # Konkatieren des alten Dataframes mit den neuen Daten
    minutenspanne = pd.date_range(start=daydata.index.min(), end=daydata.index.max(), freq='1T') # Erstellen eines Minütlichen Index
    daydata = pd.DataFrame(index=minutenspanne).join(daydata) #Zusammenführen des Index mit dem Frame
    daydata = daydata[daydata.columns[~(daydata.columns.str.contains('Unnamed') | daydata.columns.str.contains(r'\.\d'))]] #Löschen von Unnamed oder Duplizierten Spalten
    daydata=daydata[~daydata.index.duplicated(keep='last')] #Filtern von duplizierten Werten
    daydata.index.names = ['time'] #Umbenenen des Index

    merged_data=daydata.to_xarray() # Erstellen eines XArray Datasets aus dem Dataframe
    spaltennamen =["dach_CO2_ppm","dach_CO2_Sensor","dach_Diffus_CMP-11","dach_Geneigt_CM-11","dach_Global_CMP-11",
    "dach_Temp_AMUDIS_Box","dach_Temp_Bentham_Box","herrenhausen_Druck","herrenhausen_Feuchte","herrenhausen_Gust_Speed"
    ,"herrenhausen_Psychro_T","herrenhausen_Psychro_Tf","herrenhausen_Pyranometer_CM3","herrenhausen_Regen","herrenhausen_Temperatur","herrenhausen_Wind_Speed",
    "sonic_Gust_Speed","sonic_Temperatur","sonic_Wind_Dir","sonic_Wind_Speed"] #Festlegen der Spaltennamen

    # Falls ein Spaltenaneme nicht im Pandas Dataframe war wird er im Xarray Dataset als Spalte mit NaN werten hinzugefügt  
    for spaltenname in spaltennamen:
        if spaltenname  not in merged_data.data_vars:
            merged_data[spaltenname] = xr.DataArray( )  # Hinzufügen der neuen Spalte mit NaN-Werten

    merged_data.fillna(-9999)   #Verwendung von -9999 als Fillwert für Nans
    merged_data = attribute_transfer(merged_data, location=location) # Transfrerieren der Attribute für die Variablen
    merged_data.to_netcdf(filename) # Exportieren der NetCDF Datei
    return merged_data

def convert_singleday(path="/data/datenarchiv/imuk/", year="2022", month="1", day="11", location="Herrenhausen"):

    ## Funktion um einen einzelnen Tag (oder Anteil eines unvollständigen Tages) einzulesen für Herrenhausen oder Ruthe
   
    def read_and_process_data(folder, prefix, year, month, day, file_extension, tools_function, encoding="utf8"):
        ## Funktion die das Einlesen und Verarbeiten der Daten steueert
        try:
            file_path = path + folder + year + "/" + prefix + year + month.zfill(2) + day.zfill(2) + file_extension #Konstruieren des Dateipfades
            if file_extension == ".csv":
                data = pd.read_csv(file_path, delimiter=";",encoding=encoding)
            elif file_extension == ".txt":
                data = pd.read_csv(file_path, delimiter=";",encoding=encoding) 
            else:
                raise ValueError("Ungültige Datei-Erweiterung")

            data = tools_function(data) #Anwenden der Jeweiligen Toolsfunktion
            data = data.apply(pd.to_numeric, errors='coerce')


            return data
        except Exception as e: #Falls es Fehler bei den Tools Funktionen gibt, wird es nochmal mit einem anderen Datumsformat probiert, da die nicht einheitlich sind
            try: 
                data = tools_function(data, format="%d.%m.%y %H:%M:%S")
            except:
                try: 
                    data = tools_function(data, format="%d.%m.%Y %H:%M")
                except:
                    print(f"{prefix} Problem: {e}")
            return None
        
    if location== "Herrenhausen":
            # Verwendung der  read_and_process_data Funktionen um Daten für Herrenhausen,Dach und Sonic einzulesen
            herrenhausen_data = read_and_process_data("herrenhausen/", "hh", year, month, day, ".csv", herrenhausen_tools)
            dach_data = read_and_process_data("dach/", "kt", year, month, day, ".csv", dach_tools)
            sonic_data = read_and_process_data("sonic/", "sonic", year, month, day, ".txt", sonic_tools)


            # Zusammenführen der Daten
            data_list = [herrenhausen_data, dach_data, sonic_data]
            merged_data = pd.concat([data for data in data_list if data is not None], axis=1)
   
            # Filtern der Spaltennamen auf Fehler
            merged_data.columns = merged_data.columns.str.replace(r'\s*\(.*\)', '', regex=True)
            merged_data.columns = merged_data.columns.str.replace(' ', '_')


    else:
        # Verwendung der  read_and_process_data Funktionen um Daten für Ruthe und Mast einzulesen
        ruthe_data=  read_and_process_data("ruthe/", "rt", year, month, day, ".csv", ruthe_tools, encoding="latin-1")
        mast_data =read_and_process_data("ruthemast/", "rm", year, month, day, ".csv", mast_tools,encoding="latin-1")
        data_list = [ruthe_data, mast_data]
        merged_data = pd.concat([data for data in data_list if data is not None], axis=1)
        merged_data.columns = merged_data.columns.str.replace(r'\s*\(.*\)', '', regex=True)
        merged_data.columns = merged_data.columns.str.replace(' ', '_')
    return merged_data


