from netCDF4 import Dataset
import os
import xarray as xr
import glob
# Erstellen Sie eine neue NetCDF-Datei zum Zusammenfassen der Daten
def mergeyear():
    # Erstellen Sie eine leere Dataset zum Zusammenführen der Daten
    combined_dataset = xr.Dataset()

    # Iterieren Sie über die Jahre 2010-2014
    for year in range(2016, 2022):
        print(year)
        input_file = f"stunden/{year}_resample_stunden.nc"  # Annahme: Die Dateien haben das Format "datei_<jahr>.nc"

        # Überprüfen Sie, ob die Eingabedatei existiert
        if os.path.isfile(input_file):
            input_dataset = xr.open_dataset(input_file)  # Öffnen Sie die Eingabedatei

            # Führen Sie die Eingabedatei mit der kombinierten Dataset zusammen
            combined_dataset = xr.merge([combined_dataset, input_dataset])

            input_dataset.close()  # Schließen Sie die Eingabedatei

    # Speichern Sie das kombinierte Dataset in eine neue Datei
    output_file = "zusammengefasste_datei_2016-2021.nc"
    combined_dataset.to_netcdf(output_file)
    combined_dataset.close()
    return

def mergefolder(folder,output_file):
    filelist= glob.glob(folder+"/*.nc")
    combined_dataset = xr.Dataset()

    for input_file in filelist:

        # Überprüfen Sie, ob die Eingabedatei existiert
        if os.path.isfile(input_file):
            input_dataset = xr.open_dataset(input_file)  # Öffnen Sie die Eingabedatei

            # Führen Sie die Eingabedatei mit der kombinierten Dataset zusammen
            combined_dataset = xr.merge([combined_dataset, input_dataset])

            input_dataset.close()  # Schließen Sie die Eingabedatei

        # Speichern Sie das kombinierte Dataset in eine neue Datei
    #output_file = "zusammengefasste.nc"
    combined_dataset.to_netcdf(output_file)
    combined_dataset.close()
    return

#mergefolder("../Model/cortest/lstm_multi/output/all","../Visualistion/cortest_all_p.nc")
#mergefolder("../Model/timetest/lstm_multi/output/all","../Visualistion/time_test_better_a.nc")
#mergefolder("../Visualistion/AUTOARIMA","../Visualistion/auto_arima.nc")
mergefolder("../Visualistion/arma","../Visualistion/baseline.nc")
mergefolder("../Visualistion/ppp","../Visualistion/prophet.nc")


# Schließen Sie das kombinierte Dataset

