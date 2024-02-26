source /home/stadtwetter/.bashrc
#conda init bash
/home/stadtwetter/miniforge3/bin/conda activate Stadtwetter
#export PYTHONPATH=$PYTHONPATH:/localdata/weathermaps/imuk

path_input=/data/datenarchiv/imuk/
path_output=/data/stadtwetter/netcdf_daten/
#path_output=/localdata/weathermaps/webside/gross



#convert_years(path="/data/datenarchiv/imuk/", year=2022,full=True, startday="2022-01-01",endday="2022-03-01", location="Herrenhausen", filename="test_year.nc")


/home/stadtwetter/miniforge3/envs/Stadtwetter/bin/python3 /home/stadtwetter/neuralcast_webside/auto_convert.py $path_input $path_output
echo "Converting finished"


path_input=/data/stadtwetter/netcdf_daten/
path_output=/data/stadtwetter/Vorhersage/

/home/stadtwetter/miniforge3/envs/Stadtwetter/bin/python3 /home/stadtwetter/neuralcast_webside/auto_forecast.py $path_input $path_output
echo " Forecast finished"



path_input_measured=/data/stadtwetter/netcdf_daten/
path_input_forecast=/data/stadtwetter/Vorhersage/
path_output=/data/stadtwetter/Vorhersage/grafiken

/home/stadtwetter/miniforge3/envs/Stadtwetter/bin/python3 /home/stadtwetter/neuralcast_webside/auto_visualisation.py $path_input_measured $path_input_forecast $path_output
echo " Visuals finished"