import argparse
import os
from datetime import date, timedelta,datetime

from evaluation.complete_evaluations import evaluation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputpath_measured')
    parser.add_argument('inputpath_forecast_single')
    parser.add_argument('inputpath_forecast_multi')
    parser.add_argument('outputpath')

    args = parser.parse_args() 

    output =args.outputpath
    today  = datetime.now()

    evaluation()


    return

if __name__ == "__main__":
    main()