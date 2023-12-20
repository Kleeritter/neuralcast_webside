from forecast.forecast_imuknet import neural_forecast_multi,neural_forecast_single
import argparse
import os
from datetime import date, timedelta,datetime
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputpath_measured')
    parser.add_argument('inputpath_forecast')
    parser.add_argument('outputpath')

    args = parser.parse_args() 

    path = args.inputpath
    output =args.outputpath
    today  = datetime.now()

    origin =os.getcwd()
    print(origin)
    os.chdir(origin +"/neuralcast_webside")
    time_start = today.strftime('%d.%m.%Y %H:00')
    print(time_start)
    ### Imuknet1 Forecast###
    print("start Imuknet1")
    print("start Single")
    dataset= path+"/latest_herrenhausen_normal_imuknet1.nc"
    outputfile = output+"/forecast_test_single.nc"
    neural_forecast_single(dataset=dataset,outputfile=outputfile, time_start=time_start)
    #### Multi Forecast ###
    print("start Multi")
    dataset="test_data/latest_herrenhausen_normal_imuknet1.nc"
    outputfile = output+"/forecast_test.nc"
    neural_forecast_multi(dataset=dataset,outputfile=outputfile, time_start= time_start)
    #### Multi Forecast Ende ###
    
    ### Imuknet1 Forecast Ende ###


    return

if __name__ == "__main__":
    main()