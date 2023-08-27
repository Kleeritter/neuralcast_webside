# Import necessary libraries
from netCDF4 import Dataset
import os
import xarray as xr
import glob

# Define a function to merge yearly datasets
def mergeyear(Ruthe=False):
    # Create an empty combined dataset
    combined_dataset = xr.Dataset()

    # Loop through years
    for year in range(2019, 2021):
        print(year)
        if Ruthe:
            input_file = f"ruthe_{year}_resample_stunden.nc"
        else:
            input_file = f"stunden/{year}_resample_stunden.nc"

        # Check if the input file exists
        if os.path.isfile(input_file):
            input_dataset = xr.open_dataset(input_file)

            # Merge the input dataset into the combined dataset
            combined_dataset = xr.merge([combined_dataset, input_dataset])

            # Close the input dataset
            input_dataset.close()

    output_file = "zusammengefasste_datei_ruthe.nc"
    combined_dataset.to_netcdf(output_file)
    combined_dataset.close()
    return

# Define a function to merge all files in a folder
def mergefolder(folder, output_file):
    filelist = glob.glob(folder + "/*.nc")
    combined_dataset = xr.Dataset()

    # Loop through input files
    for input_file in filelist:
        if os.path.isfile(input_file):
            input_dataset = xr.open_dataset(input_file)

            # Merge the input dataset into the combined dataset
            combined_dataset = xr.merge([combined_dataset, input_dataset])

            # Close the input dataset
            input_dataset.close()

    combined_dataset.to_netcdf(output_file)
    combined_dataset.close()
    return

# Define a function to merge files in a folder while renaming variables
def merge_folder_timetest(folder, output_file, var):
    filelist = glob.glob(folder + "/*.nc")
    combined_dataset = xr.Dataset()

    # Loop through input files
    for input_file in filelist:
        if os.path.isfile(input_file):
            input_dataset = xr.open_dataset(input_file)

            # Rename the variable and use parts of the filename
            input_dataset = input_dataset.rename({var: input_file.split("/")[-1].split("_")[3] + "_" +
                                                  input_file.split("/")[-1].split("_")[4].split(".")[0] + "_" + var})
            print(input_file.split("/")[-1].split("_")[3] + "_" +
                  input_file.split("/")[-1].split("_")[4].split(".")[0] + "_" + var)

            # Merge the input dataset into the combined dataset
            combined_dataset = xr.merge([combined_dataset, input_dataset])

            # Close the input dataset
            input_dataset.close()

    combined_dataset.to_netcdf(output_file)
    combined_dataset.close()
    return

# Define a function to rename a variable in a dataset
def rename(file):
    dataset = xr.open_dataset(file).to_dataframe()
    dataset["24_24_temp"] = dataset["temp"]
    dataset.to_xarray().to_netcdf(file)
    return

# Call the 'mergefolder' function for various input folders and output files
mergefolder("../Visualistion/RUTHE_BASELINE", "../Visualistion/ruthe_baseline.nc")
mergefolder("../Visualistion/RUTHE_ARIMA", "../Visualistion/ruthe_arima.nc")
mergefolder("../Model/Ruthe/lstm_single/output/all", "../Visualistion/ruthe_lstm_single.nc")
mergefolder("../Model/Ruthe/lstm_multi/output/all", "../Visualistion/ruthe_lstm_multi.nc")
mergefolder("../Model/Ruthe/lstm_multi_red/output/all", "../Visualistion/ruthe_lstm_multi_red.nc")
mergefolder("../Model/Ruthe/lstm_multi_cor/output/all", "../Visualistion/ruthe_lstm_multi_cor.nc")

# Uncomment the following lines if needed
# mergefolder("../Model/Ruthe/lstm_multi_cor/output/all", "../Visualistion/ruthe_lstm_multi_cor.nc")
# mergeyear(Ruthe=True)

# Close the combined dataset
