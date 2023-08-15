from netCDF4 import Dataset
import os
import xarray as xr
import glob
# Erstellen Sie eine neue NetCDF-Datei zum Zusammenfassen der Daten
def mergeyear(Ruthe=False):
    # Erstellen Sie eine leere Dataset zum Zusammenführen der Daten
    combined_dataset = xr.Dataset()

    # Iterieren Sie über die Jahre 2010-2014
    for year in range(2019, 2021):
        print(year)
        if Ruthe:
            input_file = f"ruthe_{year}_resample_stunden.nc"
        else:
            input_file = f"stunden/{year}_resample_stunden.nc"  # Annahme: Die Dateien haben das Format "datei_<jahr>.nc"

        # Überprüfen Sie, ob die Eingabedatei existiert
        if os.path.isfile(input_file):
            input_dataset = xr.open_dataset(input_file)  # Öffnen Sie die Eingabedatei

            # Führen Sie die Eingabedatei mit der kombinierten Dataset zusammen
            combined_dataset = xr.merge([combined_dataset, input_dataset])

            input_dataset.close()  # Schließen Sie die Eingabedatei

    # Speichern Sie das kombinierte Dataset in eine neue Datei
    output_file = "zusammengefasste_datei_ruthe.nc"
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

def merge_folder_timetest(folder,output_file,var):
    filelist= glob.glob(folder+"/*.nc")
    combined_dataset = xr.Dataset()

    for input_file in filelist:

        # Überprüfen Sie, ob die Eingabedatei existiert
        if os.path.isfile(input_file):
            input_dataset = xr.open_dataset(input_file)  # Öffnen Sie die Eingabedatei
            input_dataset=input_dataset.rename({var:input_file.split("/")[-1].split("_")[3]+"_"+input_file.split("/")[-1].split("_")[4].split(".")[0]+"_"+var})
            print(input_file.split("/")[-1].split("_")[3]+"_"+input_file.split("/")[-1].split("_")[4].split(".")[0]+"_"+var)
            # Führen Sie die Eingabedatei mit der kombinierten Dataset zusammen
            combined_dataset = xr.merge([combined_dataset, input_dataset])

            input_dataset.close()  # Schließen Sie die Eingabedatei

        # Speichern Sie das kombinierte Dataset in eine neue Datei
    #output_file = "zusammengefasste.nc"
    combined_dataset.to_netcdf(output_file)
    combined_dataset.close()
    return

def ranme(file):
    dataset = xr.open_dataset(file).to_dataframe()
    dataset["24_24_temp"]=dataset["temp"]
    dataset.to_xarray().to_netcdf(file)
    #dataset.close()
    return#


#mergefolder("../Model/cortest/lstm_multi/output/all","../Visualistion/cortest_all_p.nc")
#mergefolder("../Model/timetest/lstm_multi/output/all","../Visualistion/time_test_better_a_n.nc")
#mergefolder("../Model/timetest/lstm_single/output/all","../Visualistion/time_test_single.nc")
#mergefolder("../Visualistion/AUTOARIMA","../Visualistion/auto_arima.nc")
#mergefolder("../Visualistion/arma","../Visualistion/baseline.nc")
#mergefolder("../Visualistion/ppp","../Visualistion/prophet.nc")
#mergefolder("../Model/Ruthe/lstm_multi/output/all","../Visualistion/ruthe_forecast.nc")

#merge_folder_timetest("../Model/timetest/lstm_multi/output/rain","../Model/timetest/lstm_multi/output/fullsets/rain_full.nc","rain")
#mergefolder("../Model/timetest/lstm_multi/output/temp","../Model/timetest/lstm_multi/output/fullsets/temp_full.nc")
#ranme("../Model/timetest/lstm_multi/output/fullsets/temp_full.nc")
#merge_folder_timetest("../Model/timetest/lstm_multi/output/winddir50sin","../Model/timetest/lstm_multi/output/fullsets/wind_dir_50_sin_full.nc","wind_dir_50_sin")
#mergefolder("../Model/baseline/output","../Model/baseline/baseline_n.nc")
mergefolder("../Visualistion/RUTHE_BASELINE","../Visualistion/ruthe_baseline.nc")
mergefolder("../Visualistion/RUTHE_ARIMA","../Visualistion/ruthe_arima.nc")
mergefolder("../Model/Ruthe/lstm_single/output/all","../Visualistion/ruthe_lstm_single.nc")
mergefolder("../Model/Ruthe/lstm_multi/output/all","../Visualistion/ruthe_lstm_multi.nc")
mergefolder("../Model/Ruthe/lstm_multi_red/output/all","../Visualistion/ruthe_lstm_multi_red.nc")

mergefolder("../Model/Ruthe/lstm_multi_cor/output/all","../Visualistion/ruthe_lstm_multi_cor.nc")
#mergefolder("../Model/Ruthe/lstm_multi_cor/output/all","../Visualistion/ruthe_lstm_multi_cor.nc")
#mergeyear(Ruthe=True)
# Schließen Sie das kombinierte Dataset

