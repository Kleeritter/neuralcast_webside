import xarray as xr
import os
import glob
# Dateinamen und Variablennamen angeben
#datei1 = '../Visualistion/best_nhit_temp.nc'
#variable1 = 'temp'

#datei2 = '../Visualistion/best_nhit_rain.nc'
#variable2 = 'rain'

# Öffnen der NetCDF-Dateien
#ds1 = xr.open_dataset(datei1)
#ds2 = xr.open_dataset(datei2)

# Extrahieren der Variablen
#var1 = ds1[variable1]
#var2 = ds2[variable2]

# Zusammenführen der Variablen
#merged = xr.merge([var1, var2])

# Speichern der zusammengeführten Datei
#merged.to_netcdf('zusammengefuegt.nc')
def mergin(folder,ouput,rename=False):
    files = glob.glob(folder+"/*.nc")
    combined_dataset = xr.Dataset()
    for file in files:

        input_dataset = xr.open_dataset(file)
        if rename:
            print(file[70:-3])
            input_dataset = input_dataset.rename({'temp':file[70:-3] })
        combined_dataset = xr.merge([combined_dataset, input_dataset])

        input_dataset.close()
        output_file = ouput
        combined_dataset.to_netcdf(output_file)
    return combined_dataset


#
mergin("/home/alex/PycharmProjects/neuralcaster/Model/timetest/lstm_multi/output/temp","../Visualistion/timetest_full_new.nc")
#mergin("/home/alex/PycharmProjects/neuralcaster/Visualistion/sarima/dart/temp","../Visualistion/timetest_sarima.nc",rename=True)