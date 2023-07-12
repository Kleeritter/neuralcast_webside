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
def mergin(folder,ouput):
    files = glob.glob(folder+"/*.nc")
    combined_dataset = xr.Dataset()
    for file in files:
        input_dataset = xr.open_dataset(file)
        combined_dataset = xr.merge([combined_dataset, input_dataset])
        input_dataset.close()
        output_file = ouput
        combined_dataset.to_netcdf(output_file)
    return combined_dataset
#varlist=["temp","rain","press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50", "wind_10", "wind_50","wind_dir_50_sin","wind_dir_50_cos"]
#combined_dataset = xr.Dataset()

# Iterieren Sie über die Jahre 2010-2014
#for var in varlist:
 #   input_file = f'../Visualistion/best_nhit_{var}.nc' # Annahme: Die Dateien haben das Format "datei_<jahr>.nc"

    # Überprüfen Sie, ob die Eingabedatei existiert
  #  if os.path.isfile(input_file):
   #     input_dataset = xr.open_dataset(input_file)  # Öffnen Sie die Eingabedatei

        # Führen Sie die Eingabedatei mit der kombinierten Dataset zusammen
    #    combined_dataset = xr.merge([combined_dataset, input_dataset])

     #   input_dataset.close()  # Schließen Sie die Eingabedatei

# Speichern Sie das kombinierte Dataset in eine neue Datei
#output_file = "../Visualistion/nhit.nc"
##-combined_dataset.to_netcdf(output_file)

# Schließen Sie das kombinierte Dataset
#combined_dataset.close()

mergin("../Visualistion/tcn","../Visualistion/tcn.nc")