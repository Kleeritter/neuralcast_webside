### Skript zur Automatischen Konvertierung von CSV in NetCDF Dateien im operativen Betrieb
### Es werden jeweils 4 NetCDF Dateien für Herrenhausen und Ruthe erstellt
### 1. Fortlaufende Jahreswerte
### 2. Fortlaufende Monatswerte
### 3. Archivieren der Monate und Jahre
### 4. Letze 72h Werte. Hauptsächlich für Neuronales Netz

from converting.netcdf_convert import *
import argparse                         # Argumentparsen für übergeordnetes Shellscript
import os
from webside_training.resample_and_normallize import resample,normalize # Resampling und Normalisieren für das Neuronale Netzwerk
from datetime import date, timedelta,datetime


def main():
    #### Argumente aus übergeorndetem Shellscript einlesen
    parser = argparse.ArgumentParser()
    parser.add_argument('inputpath')
    parser.add_argument('outputpath')
    parser.add_argument('debug')
    args = parser.parse_args() 
    path = args.inputpath

    if args.debug == 0: # Debugen eines Speziellen Datums
        today = datetime(2024, 2, 26)  # Set the desired date
    else:
        today  = date.today()

    ### 1. Erstellen der Laufenden  Jahreswerte Herrenhausen + Ruthe

    startday= today.strftime('%Y') +"-01-01"    #Anfang des aktuellen Jahres
    endday = today.strftime('%Y-%m-%d')         #Aktueller Tag
    outputpath_herrenhausen = args.outputpath+today.strftime('%Y')+"/"+today.strftime('%Y')+"_herrenhausen.nc"
    outputpath_ruthe = args.outputpath+today.strftime('%Y')+"/"+today.strftime('%Y')+"_ruthe.nc"
    # Überprüfen und Verzeichnisse erstellen
    os.makedirs(os.path.dirname(outputpath_herrenhausen), exist_ok=True)
    os.makedirs(os.path.dirname(outputpath_ruthe), exist_ok=True)
    print("start Year")
    print(startday,endday)
    origin =os.getcwd()
    os.chdir(origin +"/neuralcast_webside/converting")

    # Konvertieren der Werte
    convert_years(path=path, full=False, startday=startday,endday=endday, location="Herrenhausen", filename=outputpath_herrenhausen)
    convert_years(path=path, full=False, startday=startday,endday=endday, location="Ruthe", filename=outputpath_ruthe)

    ### 2. Erstellen der Laufenden  Monatswerte Herrenhausen + Ruthe
    print("start Month")
    startday= today.strftime('%Y-%m') +"-01"    #Anfang des aktuellen Monats
    # Überprüfen und Verzeichnisse erstellen
    outputpath_herrenhausens = args.outputpath+today.strftime('%Y')+"/"+today.strftime('%m')+"/"+today.strftime('%Y-%m')+"_herrenhausen.nc"
    outputpath_ruthes = args.outputpath+today.strftime('%Y')+"/"+today.strftime('%m')+"/"+today.strftime('%Y-%m')+"_ruthe.nc"
    os.makedirs(os.path.dirname(outputpath_herrenhausens), exist_ok=True)
    os.makedirs(os.path.dirname(outputpath_ruthes), exist_ok=True)

    # Konvertieren der Werte
    convert_years(path=path, full=False, startday=startday,endday=endday, location="Herrenhausen", filename=outputpath_herrenhausens)
    convert_years(path=path, full=False, startday=startday,endday=endday, location="Ruthe", filename=outputpath_ruthes)

    ### 3. Handling Besonderer Zeitpunkte im Jahr
    if today.strftime('%m') =="01" and today.strftime('%d') =="01" and today.strftime('%H') =="00" :#and today.strftime('%M') =="00" :
        #### Am Anfang des nächsten Jahres  wird das letzte Jahr archiviert
        print("Year completed. Start archiving")
        previos_year = today.replace(year=today.year-1)
        startday= previos_year.strftime('%Y') +"-01-01" # Anfang des Letzen Jahres
        endday = previos_year.strftime('%Y') +"-12-31"  # Ende des letzten Jahres

        # Überprüfen und Verzeichnisse erstellen
        outputpath_herrenhausen = args.outputpath+previos_year.strftime('%Y')+"/"+previos_year.strftime('%Y')+"_herrenhausen.nc"
        outputpath_ruthe = args.outputpath+previos_year.strftime('%Y')+"/"+previos_year.strftime('%Y')+"_ruthe.nc"
        os.makedirs(os.path.dirname(outputpath_herrenhausen), exist_ok=True)
        os.makedirs(os.path.dirname(outputpath_ruthe), exist_ok=True)
        
        # Konvertieren der Werte
        convert_years(path=path, full=False, startday=startday,endday=endday, location="Herrenhausen", filename=outputpath_herrenhausen)
        convert_years(path=path, full=False, startday=startday,endday=endday, location="Ruthe", filename=outputpath_ruthe)
        
        #### Am Anfang des nächsten Jahres  wird der letzte Monat des vorherigen Jahres archiviert
        previos_month = previos_year.replace(month=12)
        num_days=cal.monthrange(int(previos_month.year), int(previos_month.month))[1]
        startday= previos_month.strftime('%Y-%m') +"-01"
        endday = previos_month.strftime('%Y-%m') +"-"+str(num_days)

        # Überprüfen und Verzeichnisse erstellen
        outputpath_herrenhausens = args.outputpath+previos_month.strftime('%Y')+"/"+previos_month.strftime('%m')+"/"+previos_month.strftime('%Y-%m')+"_herrenhausen.nc"
        outputpath_ruthes = args.outputpath+previos_month.strftime('%Y')+"/"+previos_month.strftime('%m')+"/"+previos_month.strftime('%Y-%m')+"_ruthe.nc"
        os.makedirs(os.path.dirname(outputpath_herrenhausens), exist_ok=True)
        os.makedirs(os.path.dirname(outputpath_ruthes), exist_ok=True)

        # Konvertieren der Werte
        convert_years(path=path, full=False, startday=startday,endday=endday, location="Herrenhausen", filename=outputpath_herrenhausens)
        convert_years(path=path, full=False, startday=startday,endday=endday, location="Ruthe", filename=outputpath_ruthes)


    elif today.strftime('%d') =="01"  and today.strftime('%H') =="00" and today.strftime('%M') =="00":
        ### Am Anfang des nächsten Monats wird der letzte Monat archiviert 
        print("Month completed. Start archiving")
        previos_month = today.replace(month=today.month-1)
        num_days=cal.monthrange(int(previos_month.year), int(previos_month.month))[1]
        startday= previos_month.strftime('%Y-%m') +"-01" #Erster Tag des Vorangeganenen Monats
        endday = previos_month.strftime('%Y-%m') +"-"+str(num_days) # Letzter Tag des vorangeganenen Monats

        # Überprüfen und Verzeichnisse erstellen
        outputpath_herrenhausens = args.outputpath+previos_month.strftime('%Y')+"/"+previos_month.strftime('%m')+"/"+previos_month.strftime('%Y-%m')+"_herrenhausen.nc"
        outputpath_ruthes = args.outputpath+previos_month.strftime('%Y')+"/"+previos_month.strftime('%m')+"/"+previos_month.strftime('%Y-%m')+"_ruthe.nc"
        os.makedirs(os.path.dirname(outputpath_herrenhausens), exist_ok=True)
        os.makedirs(os.path.dirname(outputpath_ruthes), exist_ok=True)

        # Konvertieren der Werte
        convert_years(path=path, full=False, startday=startday,endday=endday, location="Herrenhausen", filename=outputpath_herrenhausens)
        #convert_years(path=path, full=False, startday=startday,endday=endday, location="Ruthe", filename=outputpath_ruthes)


    else:
        pass


    ### 4. Werte der Letzten 72h Stunden (Wichtig für das Neuronale Netzwerk)

    print ("start 72 cycle")
    startday= (today - timedelta(hours=72)).strftime('%Y-%m-%d') #Start ist Anfang des Tages der 72h Zurückliegt
    endday = today.strftime('%Y-%m-%d') #Ende ist Anfang des heutigen Tages
    outputpath_herrenhausen = args.outputpath+"latest_herrenhausen.nc"
    outputpath_ruthe = args.outputpath+"latest_ruthe.nc"

    # Überprüfen und Verzeichnisse erstellen
    os.makedirs(os.path.dirname(outputpath_herrenhausen), exist_ok=True)
    os.makedirs(os.path.dirname(outputpath_ruthe), exist_ok=True)

    # Konvertieren der Werte
    convert_years(path=path, full=False, startday=startday,endday=endday, location="Herrenhausen", filename=outputpath_herrenhausen)
    #convert_years(path=path, full=False, startday=startday,endday=endday, location="Ruthe", filename=outputpath_ruthe)



    ### 5. Resamplen auf Stundenwerte und Normalisieren für das Neuronale Netzwerk

    print("start resampling imuknet1")
    outputpath_herrenhausen_res = args.outputpath+"latest_herrenhausen_res_imuknet1.nc"
    outputpath_herrenhausen_normal = args.outputpath+"latest_herrenhausen_normal_imuknet1.nc"
    outputpath_ruthe_res = args.outputpath+"latest_ruthe_res_imuknet1.nc"
    outputpath_ruthe_normal = args.outputpath+"latest_ruth_normal_imuknet1.nc"

    resample(outputpath_herrenhausen, outputpath_herrenhausen_res,v=1)
    normalize(outputpath_herrenhausen_res,outputpath_herrenhausen_normal,v=1)
    #resample(outputpath_ruthe, outputpath_ruthe_res,v=1)
    #normalize(outputpath_ruthe_res,outputpath_ruthe_normal,v=1)

    print("start resampling imuknet2")
    outputpath_herrenhausen_res = args.outputpath+"latest_herrenhausen_res_imuknet2.nc"
    outputpath_herrenhausen_normal = args.outputpath+"latest_herrenhausen_normal_imuknet2.nc"
    outputpath_ruthe_res = args.outputpath+"latest_ruthe_res_imuknet2.nc"
    outputpath_ruthe_normal = args.outputpath+"latest_ruth_normal_imuknet2.nc"

    resample(outputpath_herrenhausen, outputpath_herrenhausen_res,v=2)
    normalize(outputpath_herrenhausen_res,outputpath_herrenhausen_normal,v=2)
    #resample(outputpath_ruthe, outputpath_ruthe_res,v=2)
    #normalize(outputpath_ruthe_res,outputpath_ruthe_normal,v=2)

if __name__ == "__main__":
    main()