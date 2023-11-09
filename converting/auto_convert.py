from netcdf_convert import *
import argparse

from datetime import date
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
    parser.add_argument('outputpath_herrenhausen')
    parser.add_argument('outputpath_ruthe')

    args = parser.parse_args()  # gv[480#210    #480
    #year = int(args.year)
    #if args.full == "True":
     #   full= True
    #else:
     #   full=False

    path = args.inputpath
    today = date.today()
    startday= today.strftime('%Y') +"-01-01"
    endday = today.strftime('%Y-%m-%d')
    outputpath_herrenhausen = args.outputpath_herrenhausen + startday+"-"+endday+"_herrenhausen.nc"
    outputpath_ruthe = args.outputpath_ruthe+ startday+"-"+endday+"_ruthe.nc"
    #location=args.location
    #os.chdir(dir_Produkt)
    #convert_years(path="/data/datenarchiv/imuk/", year=2022,full=True, startday="2022-01-01",endday="2022-03-01", location="Herrenhausen", filename="test_year.nc")
    convert_years(path=path, year=year,full=False, startday=startday,endday=endday, location="Herrenhausen", filename=outputpath_herrenhausen)
    convert_years(path=path, year=year,full=False, startday=startday,endday=endday, location="Ruthe", filename=outputpath_ruthe)


if __name__ == "__main__":
    main()