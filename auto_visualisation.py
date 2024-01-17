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

    args = parser.parse_args() 

    output =args.outputpath
    today  = datetime.now()

    origin =os.getcwd()
    print(origin)
    os.chdir(origin +"/neuralcast_webside")
    #time_start = today.strftime('%d.%m.%Y %H:00')
    print(time_start)
    ### Imuknet1 Forecast###
    print("start Imuknet1")
  

    forecast_vars=["herrenhausen_Temperatur","derived_Press_sl","herrenhausen_Feuchte","dach_Diffus_CMP-11","dach_Global_CMP-11","herrenhausen_Gust_Speed", "sonic_Gust_Speed","herrenhausen_Regen","herrenhausen_Wind_Speed",
       "sonic_Wind_Speed","sonic_Wind_Dir_sin","sonic_Wind_Dir_cos","derived_Taupunkt","derived_Taupunkt3h","derived_Press3h", "derived_rainsum3h","derived_Temp3h","derived_vertwind" ]
    
    for forecast_var in forecastvars:
        visualize_var(forecast_var=forecast_var,measured_data_path=args.inputpath_measured, forecast_multi_path=args.inputpath_forecast_multi,forecast_single_path=args.inputpath_forecast_single,outputpath=args.outputpath)



    return

if __name__ == "__main__":
    main()