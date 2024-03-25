from webside_training.resample_and_normallize import resample, normalize
import glob
import xarray as xr 
import os


def merge(ordner_pfad):

    import xarray as xr
    import os

    # Pfad zum Ordner mit den NetCDF-Dateien
    #ordner_pfad = '/pfad/zum/deinem/ordner/'
    dateien = glob.glob(ordner_pfad+"/*.nc")

    ds = xr.open_dataset(dateien[0])
    for datei in dateien[1:]:
        print(datei)
        ds_tp = xr.open_dataset(datei)
        ds = ds.merge(ds_tp)
        print(clonk)
        ds_tp.close()

    # Speichern Sie den kombinierten Datensatz in einer neuen Datei
    ds.to_netcdf('datei.nc')


    return
#merge("/mnt/nvmente/CODE/neuralcast_webside/webside_training/Archiv")


def ress(ordner_pfad,outpath):
    dateien = [f for f in os.listdir(ordner_pfad) if f.endswith('.nc')]
    for datei in dateien:
        print(datei)
        print(os.path.join(outpath, datei))
        resample(os.path.join(ordner_pfad, datei),os.path.join(outpath, datei),v=1)
    return
#ress ("/mnt/nvmente/CODE/neuralcast_webside/webside_training/Archiv","/mnt/nvmente/CODE/neuralcast_webside/webside_training/Archiv/res")


def mergefolder(folder, output_file):
    filelist = glob.glob(folder + "/*.nc")
    print(filelist)
    combined_dataset = xr.Dataset()

    # Loop through input files
    for input_file in filelist:
        print(input_file)
        if os.path.isfile(input_file):
            input_dataset = xr.open_dataset(input_file)

            # Merge the input dataset into the combined dataset
            combined_dataset = xr.merge([combined_dataset, input_dataset])

            # Close the input dataset
            input_dataset.close()

    combined_dataset.to_netcdf(output_file)
    combined_dataset.close()
    return

#mergefolder("/mnt/nvmente/CODE/neuralcast_webside/webside_training/Archiv/res", "merger.nc")


def normalsss(input,outpath):

    normalize(input,outpath,v=1)
    return
normalsss("/mnt/nvmente/CODE/neuralcast_webside/webside_training/Archiv/merger.nc", "/mnt/nvmente/CODE/neuralcast_webside/webside_training/Archiv/normal.nc")