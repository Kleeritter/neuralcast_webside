#source /home/stadtwetter/.bashrc
#conda init
/home/alex/miniforge3/bin/conda activate Stadtwetter
#export PYTHONPATH=$PYTHONPATH:/localdata/weathermaps/imuk

path_root=/mnt/nvmente/CODE/neuralcast_webside

path_input="${path_root}/testdata/datenarchiv/imuk/"
path_output="${path_root}/testdata/stadtwetter/netcdf_daten/"

cd /mnt/nvmente/CODE
#convert_years(path="/data/datenarchiv/imuk/", year=2022,full=True, startday="2022-01-01",endday="2022-03-01", location="Herrenhausen", filename="test_year.nc")

#/home/alex/miniforge3/envs/Stadtwetter/bin/python3 "${path_root}/auto_convert.py" $path_input $path_output
#echo "Converting finished"


path_input_forecast="${path_root}/testdata/stadtwetter/netcdf_daten/"
path_output_forecast="${path_root}/testdata/stadtwetter/Vorhersage/"

/home/alex/miniforge3/envs/Stadtwetter/bin/python3 "${path_root}/auto_forecast.py" $path_input_forecast $path_output_forecast
echo " Forecast finished"



path_input_measured=/data/stadtwetter/netcdf_daten/
path_input_forecast=/data/stadtwetter/Vorhersage/
path_output=/data/stadtwetter/Vorhersage/grafiken

#/home/alex/miniforge3/envs/Stadtwetter/bin/python3 /home/stadtwetter/neuralcast_webside/auto_visualisation.py $path_input_measured $path_input_forecast $path_output
#echo " Visuals finished"