from netcdf_convert import *
import argparse
def main():
    ##Parsing Variable Values
    ##Parsing Variable Values
    parser = argparse.ArgumentParser()
    parser.add_argument('inputpath')
    parser.add_argument('year')  #
    parser.add_argument('full')
    parser.add_argument('startday')
    parser.add_argument('endday')
    parser.add_argument('location')
    parser.add_argument('outputpath')

    args = parser.parse_args()  # gv[480#210    #480
    year = int(args.year)
    if args.full == "True":
        full= True
    else:
        full=False
    filename = args.outputpath
    path = args.inputpath

    startday = args.startday
    endday = args.endday
    location=args.location
    #os.chdir(dir_Produkt)
    #convert_years(path="/data/datenarchiv/imuk/", year=2022,full=True, startday="2022-01-01",endday="2022-03-01", location="Herrenhausen", filename="test_year.nc")
    convert_years(path=path, year=year,full=full, startday=startday,endday=endday, location=location, filename=filename)


if __name__ == "__main__":
    main()