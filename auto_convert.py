from converting.netcdf_convert import *
import argparse
import os
from webside_training.resample_and_normallize import resample,normalize

from datetime import date, timedelta,datetime
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('inputpath')
    parser.add_argument('outputpath')

    args = parser.parse_args() 

    path = args.inputpath
    today  = date.today()
    
    startday= today.strftime('%Y') +"-01-01"
    endday = today.strftime('%Y-%m-%d')
    outputpath_herrenhausen = args.outputpath+today.strftime('%Y')+"/"+today.strftime('%Y')+"_herrenhausen.nc"
    outputpath_ruthe = args.outputpath+today.strftime('%Y')+"/"+today.strftime('%Y')+"_ruthe.nc"
    # Überprüfen und Verzeichnisse erstellen
    os.makedirs(os.path.dirname(outputpath_herrenhausen), exist_ok=True)
    os.makedirs(os.path.dirname(outputpath_ruthe), exist_ok=True)
    print("start Year")
    origin =os.getcwd()
    print(origin)
    os.chdir(origin +"/neuralcast_webside/converting")
    #convert_years(path="/data/datenarchiv/imuk/", year=2022,full=True, startday="2022-01-01",endday="2022-03-01", location="Herrenhausen", filename="test_year.nc")
    convert_years(path=path, full=False, startday=startday,endday=endday, location="Herrenhausen", filename=outputpath_herrenhausen)
    convert_years(path=path, full=False, startday=startday,endday=endday, location="Ruthe", filename=outputpath_ruthe)

    print("start Month")
    startday= today.strftime('%Y-%m') +"-01"
    outputpath_herrenhausens = args.outputpath+today.strftime('%Y')+"/"+today.strftime('%m')+"/"+today.strftime('%Y-%m')+"_herrenhausen.nc"
    outputpath_ruthes = args.outputpath+today.strftime('%Y')+"/"+today.strftime('%m')+"/"+today.strftime('%Y-%m')+"_ruthe.nc"
    os.makedirs(os.path.dirname(outputpath_herrenhausens), exist_ok=True)
    os.makedirs(os.path.dirname(outputpath_ruthes), exist_ok=True)
    convert_years(path=path, full=False, startday=startday,endday=endday, location="Herrenhausen", filename=outputpath_herrenhausens)
    convert_years(path=path, full=False, startday=startday,endday=endday, location="Ruthe", filename=outputpath_ruthes)

    if today.strftime('%m') =="01" and today.strftime('%d') =="01" and today.strftime('%H') =="00" and today.strftime('%M') =="00" :
        print("Year completed. Start archiving")
        previos_year = today.replace(year=today.year-1)
        startday= previos_year.strftime('%Y') +"-01-01"
        endday = previos_year.strftime('%Y') +"-12-31"
        outputpath_herrenhausen = args.outputpath+previos_year.strftime('%Y')+"/"+previos_year.strftime('%Y')+"_herrenhausen.nc"
        outputpath_ruthe = args.outputpath+previos_year.strftime('%Y')+"/"+previos_year.strftime('%Y')+"_ruthe.nc"
        os.makedirs(os.path.dirname(outputpath_herrenhausen), exist_ok=True)
        os.makedirs(os.path.dirname(outputpath_ruthe), exist_ok=True)
        convert_years(path=path, full=False, startday=startday,endday=endday, location="Herrenhausen", filename=outputpath_herrenhausen)
        convert_years(path=path, full=False, startday=startday,endday=endday, location="Ruthe", filename=outputpath_ruthe)
        
        
        previos_month = previos_year.replace(month=12)
        num_days=cal.monthrange(int(previos_month.year), int(previos_month.month))[1]
        startday= previos_month.strftime('%Y-%m') +"-01"
        endday = previos_month.strftime('%Y-%m') +"-"+str(num_days)
        outputpath_herrenhausens = args.outputpath+previos_month.strftime('%Y')+"/"+previos_month.strftime('%m')+"/"+previos_month.strftime('%Y-%m')+"_herrenhausen.nc"
        outputpath_ruthes = args.outputpath+previos_month.strftime('%Y')+"/"+previos_month.strftime('%m')+"/"+previos_month.strftime('%Y-%m')+"_ruthe.nc"
        os.makedirs(os.path.dirname(outputpath_herrenhausens), exist_ok=True)
        os.makedirs(os.path.dirname(outputpath_ruthes), exist_ok=True)
        convert_years(path=path, full=False, startday=startday,endday=endday, location="Herrenhausen", filename=outputpath_herrenhausens)
        convert_years(path=path, full=False, startday=startday,endday=endday, location="Ruthe", filename=outputpath_ruthes)


    elif today.strftime('%d') =="01"  and today.strftime('%H') =="00" and today.strftime('%M') =="00":
        print("Month completed. Start archiving")
        previos_month = today.replace(month=today.month-1)
        num_days=cal.monthrange(int(previos_month.year), int(previos_month.month))[1]
        startday= previos_month.strftime('%Y-%m') +"-01"
        endday = previos_month.strftime('%Y-%m') +"-"+str(num_days)
        outputpath_herrenhausens = args.outputpath+previos_month.strftime('%Y')+"/"+previos_month.strftime('%m')+"/"+previos_month.strftime('%Y-%m')+"_herrenhausen.nc"
        outputpath_ruthes = args.outputpath+previos_month.strftime('%Y')+"/"+previos_month.strftime('%m')+"/"+previos_month.strftime('%Y-%m')+"_ruthe.nc"
        os.makedirs(os.path.dirname(outputpath_herrenhausens), exist_ok=True)
        os.makedirs(os.path.dirname(outputpath_ruthes), exist_ok=True)
        convert_years(path=path, full=False, startday=startday,endday=endday, location="Herrenhausen", filename=outputpath_herrenhausens)
        convert_years(path=path, full=False, startday=startday,endday=endday, location="Ruthe", filename=outputpath_ruthes)


    else:
        pass

    print ("start 72 cycle")
    startday= (today - timedelta(hours=72)).strftime('%Y-%m-%d')
    endday = today.strftime('%Y-%m-%d')
    outputpath_herrenhausen = args.outputpath+"latest_herrenhausen.nc"
    outputpath_ruthe = args.outputpath+"latest_ruthe.nc"
    # Überprüfen und Verzeichnisse erstellen
    os.makedirs(os.path.dirname(outputpath_herrenhausen), exist_ok=True)
    os.makedirs(os.path.dirname(outputpath_ruthe), exist_ok=True)
    convert_years(path=path, full=False, startday=startday,endday=endday, location="Herrenhausen", filename=outputpath_herrenhausen)
    convert_years(path=path, full=False, startday=startday,endday=endday, location="Ruthe", filename=outputpath_ruthe)

    print("start resampling imuknet1")
    outputpath_herrenhausen_res = args.outputpath+"latest_herrenhausen_res_imuknet1.nc"
    outputpath_herrenhausen_normal = args.outputpath+"latest_herrenhausen_normal_imuknet1.nc"
    outputpath_ruthe_res = args.outputpath+"latest_ruthe_res_imuknet1.nc"
    outputpath_ruthe_normal = args.outputpath+"latest_ruth_normal_imuknet1.nc"

    resample(outputpath_herrenhausen, outputpath_herrenhausen_res,v=1)
    normalize(outputpath_herrenhausen_res,outputpath_herrenhausen_normal,v=1)
    resample(outputpath_ruthe, outputpath_ruthe_res,v=1)
    normalize(outputpath_ruthe_res,outputpath_ruthe_normal,v=1)

    print("start resampling imuknet2")
    outputpath_herrenhausen_res = args.outputpath+"latest_herrenhausen_res_imuknet2.nc"
    outputpath_herrenhausen_normal = args.outputpath+"latest_herrenhausen_normal_imuknet2.nc"
    outputpath_ruthe_res = args.outputpath+"latest_ruthe_res_imuknet2.nc"
    outputpath_ruthe_normal = args.outputpath+"latest_ruth_normal_imuknet2.nc"

    resample(outputpath_herrenhausen, outputpath_herrenhausen_res,v=2)
    normalize(outputpath_herrenhausen_res,outputpath_herrenhausen_normal,v=2)
    resample(outputpath_ruthe, outputpath_ruthe_res,v=2)
    normalize(outputpath_ruthe_res,outputpath_ruthe_normal,v=2)

if __name__ == "__main__":
    main()