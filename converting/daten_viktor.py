from netcdf_convert import *
import argparse
import numpy as np
import os
def main():
    ##Parsing Variable Values
    ##Parsing Variable Values
    os.chdir("/Users/alex/Code/neuralcast_webside/converting")
    years = np.arange(2008,2023)
    path = "/Users/alex/Downloads/herrenhausen/"
    for year in years:
        print(year)
        filename = "viktor/"+str(year)+".nc"
        convert_years(path=path, year=year,full=True, filename=filename)


if __name__ == "__main__":
    main()