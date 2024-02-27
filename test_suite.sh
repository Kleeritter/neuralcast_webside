#source /home/stadtwetter/.bashrc
#conda init
#export PYTHONPATH=$PYTHONPATH:/localdata/weathermaps/imuk

#root_root=/Users/alex/Code #/mnt/nvmente/CODE
root_root=/mnt/nvmente/CODE

#python_root=/Users/alex #/home/alex
python_root=/home/alex
#"${python_root}/miniforge3/bin/conda" init zsh
"${python_root}/miniforge3/bin/conda" activate Stadtwetter


path_root="${root_root}/neuralcast_webside"

path_input="${path_root}/testdata/datenarchiv/imuk/"
path_output="${path_root}/testdata/stadtwetter/netcdf_daten/"

#debug=True

cd $root_root
#convert_years(path="/data/datenarchiv/imuk/", year=2022,full=True, startday="2022-01-01",endday="2022-03-01", location="Herrenhausen", filename="test_year.nc")

"${python_root}/miniforge3/envs/Stadtwetter/bin/python3" "${path_root}/auto_convert.py" $path_input $path_output
echo "Converting finished"

#cd $root_root
path_input_forecast="${path_root}/testdata/stadtwetter/netcdf_daten/"
path_output_forecast="${path_root}/testdata/stadtwetter/Vorhersage/"

"${python_root}/miniforge3/envs/Stadtwetter/bin/python3" "${path_root}/auto_forecast.py" $path_input_forecast $path_output_forecast
echo " Forecast finished"



path_input_measured="${path_root}/testdata/stadtwetter/netcdf_daten/latest_herrenhausen_res_imuknet1.nc"
path_input_forecast_single="${path_root}/testdata/stadtwetter/Vorhersage/forecast_test_single.nc"
path_input_forecast_multi="${path_root}/testdata/stadtwetter/Vorhersage/forecast_test.nc"

path_output="${path_root}/testdata/stadtwetter/Vorhersage/grafiken/"


"${python_root}/miniforge3/envs/Stadtwetter/bin/python3" "${path_root}/auto_visualisation.py" $path_input_measured $path_input_forecast_single $path_input_forecast_multi $path_output
echo " Visuals finished"