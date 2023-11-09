from netcdf_convert import *
import argparse
import os

from datetime import date, timedelta
def main():
    ##Parsing Variable Values
    ##Parsing Variable Values
    parser = argparse.ArgumentParser()
    #parser.add_argument("")
    parser.add_argument('inputpath')
    #parser.add_argument('year')  #
    #parser.add_argument('full')
    #parser.add_argument('startday')
    #parser.add_argument('endday')
    #parser.add_argument('location')
    parser.add_argument('outputpath')


    args = parser.parse_args()  # gv[480#210    #480
    #year = int(args.year)
    #if args.full == "True":
     #   full= True
    #else:
     #   full=False

    path = args.inputpath
    today = date_object = datetime.strptime("2023-01-01", '%Y-%m-%d')
    #date.today()
    startday= today.strftime('%Y') +"-01-01"
    endday = today.strftime('%Y-%m-%d')
    outputpath_herrenhausen = args.outputpath+today.strftime('%Y')+"/"+today.strftime('%Y')+"_ongoning_herrenhausen.nc"
    outputpath_ruthe = args.outputpath+today.strftime('%Y')+"/"+today.strftime('%Y')+"_ongoning_ruthe.nc"
    # Überprüfen und Verzeichnisse erstellen
    os.makedirs(os.path.dirname(outputpath_herrenhausen), exist_ok=True)
    os.makedirs(os.path.dirname(outputpath_ruthe), exist_ok=True)

    #convert_years(path="/data/datenarchiv/imuk/", year=2022,full=True, startday="2022-01-01",endday="2022-03-01", location="Herrenhausen", filename="test_year.nc")
    convert_years(path=path, full=False, startday=startday,endday=endday, location="Herrenhausen", filename=outputpath_herrenhausen)
    convert_years(path=path, full=False, startday=startday,endday=endday, location="Ruthe", filename=outputpath_ruthe)

    if today.strftime('%m') =="01" and today.strftime('%d') =="01" :
        print("Year completed. Start archiving")
        previos_year = today.replace(year=current.year-1)
        startday= previos_year.strftime('%Y') +"-01-01"
        endday = previos_year.strftime('%Y') +"-12-31"
        outputpath_herrenhausen = args.outputpath+today.strftime('%Y')+"/"+today.strftime('%Y')+"_herrenhausen.nc"
        outputpath_ruthe = args.outputpath+today.strftime('%Y')+"/"+today.strftime('%Y')+"_ruthe.nc"
        convert_years(path=path, full=False, startday=startday,endday=endday, location="Herrenhausen", filename=outputpath_herrenhausen)
        convert_years(path=path, full=False, startday=startday,endday=endday, location="Ruthe", filename=outputpath_ruthe)
        previos_month = today.replace(monht=current.month-1)
        num_days=cal.monthrange(int(previos_month.year), int(previos_month.month))[1]
        startday= previos_month.strftime('%Y-%m') +"-01"
        endday = previos_month.strftime('%Y-%m') +str(num_days)
        outputpath_herrenhausen = args.outputpath+previos_month.strftime('%Y')+"/"+previos_month.strftime('%m')+"/"+previos_month.strftime('%Y-%d')+"_herrenhausen.nc"
        outputpath_herrenhausen = args.outputpath+previos_month.strftime('%Y')+"/"+previos_month.strftime('%m')+"/"+previos_month.strftime('%Y-%d')+"_ruthe.nc"
        os.makedirs(os.path.dirname(outputpath_herrenhausen), exist_ok=True)
        os.makedirs(os.path.dirname(outputpath_ruthe), exist_ok=True)
        convert_years(path=path, full=False, startday=startday,endday=endday, location="Herrenhausen", filename=outputpath_herrenhausen)
        convert_years(path=path, full=False, startday=startday,endday=endday, location="Ruthe", filename=outputpath_ruthe)


    elif today.strftime('%d') =="01" :
        print("Month completed. Start archiving")
        previos_month = today.replace(monht=current.month-1)
        num_days=cal.monthrange(int(previos_month.year), int(previos_month.month))[1]
        startday= previos_month.strftime('%Y-%m') +"-01"
        endday = previos_month.strftime('%Y-%m') +str(num_days)
        outputpath_herrenhausen = args.outputpath+previos_month.strftime('%Y')+"/"+previos_month.strftime('%m')+"/"+previos_month.strftime('%Y-%d')+"_herrenhausen.nc"
        outputpath_herrenhausen = args.outputpath+previos_month.strftime('%Y')+"/"+previos_month.strftime('%m')+"/"+previos_month.strftime('%Y-%d')+"_ruthe.nc"
        os.makedirs(os.path.dirname(outputpath_herrenhausen), exist_ok=True)
        os.makedirs(os.path.dirname(outputpath_ruthe), exist_ok=True)
        convert_years(path=path, full=False, startday=startday,endday=endday, location="Herrenhausen", filename=outputpath_herrenhausen)
        convert_years(path=path, full=False, startday=startday,endday=endday, location="Ruthe", filename=outputpath_ruthe)

    else:
        pass




if __name__ == "__main__":
    main()