from forecast.forecast_imuknet import neural_forecast_multi,neural_forecast_single
import argparse
import os
from datetime import date, timedelta,datetime
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputpath')
    parser.add_argument('outputpath')
    parser.add_argument('debug')
    args = parser.parse_args() 

    path = args.inputpath
    output =args.outputpath
    debug = int(args.debug)
    print(debug)
    #debug=0
    if debug==0:
        print("DEBUG")
        today = datetime(2024, 2, 25,3)  # Set the desired date
    else:
        today  = date.today()

    origin =os.getcwd()
    print(origin)
    os.chdir(origin +"/neuralcast_webside")
    time_start = today.strftime('%d.%m.%Y %H:00')
    print(time_start)
    print(today)
    ### Imuknet1 Forecast###
    print("start Imuknet1")
    print("start Single")
    dataset= path+"/latest_herrenhausen_normal_imuknet1.nc"
    outputfile = output+"/forecast_test_single_multiday.nc"
    neural_forecast_single(dataset=dataset,outputfile=outputfile, time_start=time_start, today=today)
    #### Multi Forecast ###
    print("start Multi")
    dataset=path+"/latest_herrenhausen_normal_imuknet1.nc"
    outputfile = output+"/forecast_test.nc"
    neural_forecast_multi(dataset=dataset,outputfile=outputfile, time_start= time_start)
    #### Multi Forecast Ende ###
    
    ### Imuknet1 Forecast Ende ###


    return

if __name__ == "__main__":
    main()