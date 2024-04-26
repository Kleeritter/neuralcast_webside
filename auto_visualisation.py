import argparse
import os
from datetime import date, timedelta,datetime

from visuals.complete_visual import visualize_var

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputpath_measured')
    parser.add_argument('inputpath_forecast_single')
    parser.add_argument('inputpath_forecast_multi')
    parser.add_argument('outputpath')
    parser.add_argument('debug')
   # parser.add_argument('insert_test_path')


    args = parser.parse_args() 

    output =args.outputpath
    today  = datetime.now()
    debug = int(args.debug)
    #input_test =args.insert_test_path
    print(debug)
    if debug==0:
        print("DEBUG")
        debug=True
        today = datetime(2024, 2, 27,2)  # Set the desired date
    else:
        today  = date.today()
        debug=False

    #origin =os.getcwd()
    #print(origin)
    #os.chdir(origin +"/neuralcast_webside")
    #time_start = today.strftime('%d.%m.%Y %H:00')
    #print(time_start)
    ### Imuknet1 Forecast###
    print("start Imuknet1")
  

    forecast_vars=["herrenhausen_Temperatur","derived_Press_sl","herrenhausen_Feuchte","dach_Diffus_CMP-11","dach_Global_CMP-11","herrenhausen_Gust_Speed", "sonic_Gust_Speed","herrenhausen_Regen","herrenhausen_Wind_Speed",
       "sonic_Wind_Speed","sonic_Wind_Dir"]
    
    for forecast_var in forecast_vars:
        visualize_var(forecast_var=forecast_var,measured_data_path=args.inputpath_measured, forecast_multi_path=args.inputpath_forecast_multi,forecast_single_path=args.inputpath_forecast_single,outputpath=args.outputpath,debug=debug)
        print(forecast_var+ " completed")

    return

if __name__ == "__main__":
    main()