
def collect_forecasts(inputpath_measured, inputpath_forecast_multi, inputpath_forecast_single, outputpath_collected, now):

    from os.path import exists
    
    
    import xarray as xr
    input_measured = xr.open_dataset(inputpath_measured)
    input_forecast_single = xr.open_dataset(inputpath_forecast_single)
    # Umbenennung der Variablen
    for variable in input_forecast_single.variables:
            if variable != "time":
                new_variable_name = "imuknet1_single_" + variable
                input_forecast_single = input_forecast_single.rename({variable: new_variable_name})

    input_forecast_multi = xr.open_dataset(inputpath_forecast_multi)

    # Umbenennung der Variablen
    for variable in input_forecast_multi.variables:
            if variable != "time":
                new_variable_name = "imuknet1_multi_" + variable
                input_forecast_multi = input_forecast_multi.rename({variable: new_variable_name})

    merged_data = xr.merge([input_measured, input_forecast_single, input_forecast_multi])

    file_exists = exists(outputpath_collected)

    if file_exists == True:
            Data = xr.open_dataset(outputpath_collected)
            merged_data = xr.merge([merged_data, Data])


    merged_data.to_netcdf(outputpath_collected)



    return




def evaluation(inputpath_measured = "", inputpath_forecast_multi = "", inputpath_forecast_single = "", outputpath_collected=""):
    import datetime
    import os
    import xarray as xr
    now = datetime.datetime.now()

    collect_forecasts(inputpath_measured=inputpath_measured,
    inputpath_forecast_multi=inputpath_forecast_multi,inputpath_forecast_single=inputpath_forecast_single,
    outputpath_collected=outputpath_collected, now=now)

    return


if __name__ == "__main__":
    evaluation(inputpath_measured= "/Users/alex/Code/gong+/Testdaten/roling_rmse/latest_herrenhausen_res_imuknet1.nc",
    inputpath_forecast_multi= "/Users/alex/Code/gong+/Testdaten/roling_rmse/forecast_test.nc", inputpath_forecast_single= "/Users/alex/Code/gong+/Testdaten/roling_rmse/forecast_test_single.nc",
    outputpath_collected = "/Users/alex/Code/gong+/Testdaten/roling_rmse/hans.nc")
