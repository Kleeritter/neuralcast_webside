## Archive 00 and 12h runs from the neural_models for later evaulations


import argparse       
import xarray as xr
import os
import pandas as pd
import yaml

def main():
    #### Argumente aus übergeorndetem Shellscript einlesen
    parser = argparse.ArgumentParser()
    parser.add_argument('inputpath_forecast_single')
    parser.add_argument('inputpath_forecast_multi')
    parser.add_argument('ongoing_file')
    parser.add_argument('debug')
    args = parser.parse_args() 
    path_single = args.inputpath_forecast_single
    path_multi = args.inputpath_forecast_multi
    path_ongoing = args.ongoing_file

    try:
        single = xr.open_dataset(path_single)
    except Exception as er:
        print("could not read in Single", er)

    try:
        multi = xr.open_dataset(path_multi)
    except Exception as er:
        print("could not read in multi", er)


    try: 
        ongoing_dataset = xr.open_dataset(path_ongoing)
        os.remove(path_ongoing)
        newgoing=False
    except Exception as er:
        print("could not read in ongoing", er)
        print("creating new file")
        ongoing_dataset=xr.Dataset()
        newgoing=True

    def filter_dataset(ds):
    # Extrahiere alle Variablennamen aus dem Dataset
        variable_names = list(ds.data_vars.keys())

        filtered_variable_names = [var for var in variable_names if var.endswith(("00", "06"))]


        # Extrahiere die entsprechenden Variablen aus dem Dataset
        filtered_variables = ds[filtered_variable_names]
        print(filtered_variables)
        return  filtered_variables
    
    single_filterd = filter_dataset(single).to_dataframe()
    multi_filterd = filter_dataset(multi).to_dataframe()

    merged_multi= pd.merge(single_filterd, multi_filterd, on='time',how='outer', suffixes=('_single', '_multi',))

    print(merged_multi)
    
    if newgoing:
        merged_ongoing =merged_multi
    else:
        #merged_ongoing= pd.merge(ongoing_dataset.to_dataframe(), merged_multi, on='time',how='inner')#
        merged_ongoing = pd.concat([ongoing_dataset.to_dataframe(), merged_multi]).drop_duplicates()
    
    print(merged_ongoing)
    print(os.getcwd())
    ongoing_dataset = merged_ongoing.to_xarray()

    with open("neuralcast_webside/archiving/model_attributes.yaml", 'r') as yaml_file:
        attribute_data = yaml.safe_load(yaml_file)

    # Aktualisieren der Attribute der einzelnen Variablen
    vars= ongoing_dataset.keys()
    for var in vars:
        parts = var.split('_')
        varo = '_'.join(parts[:2])
        print(varo)
        if varo in attribute_data:
            for key, value in attribute_data[varo].items():
                ongoing_dataset[var].attrs[key] = value
    print(ongoing_dataset)
    ongoing_dataset.to_netcdf(path_ongoing)
    return



if __name__ == "__main__":
    main()
