source /home/stadtwetter/.bashrc
#conda init
#export PYTHONPATH=$PYTHONPATH:/localdata/weathermaps/imuk

root_root=/home/stadtwetter #/mnt/nvmente/CODE
#root_root=/mnt/nvmente/CODE

python_root=/home/stadtwetter #/home/alex
#python_root=/home/alex
#"${python_root}/miniforge3/bin/conda" init zsh
"${python_root}/miniforge3/bin/conda" activate Stadtwetter


path_root="${root_root}/neuralcast_webside"

path_input="/data/datenarchiv/imuk/"
path_output="/data/stadtwetter/netcdf_daten/"

debug=1  # true
cd $root_root
#convert_years(path="/data/datenarchiv/imuk/", year=2022,full=True, startday="2022-01-01",endday="2022-03-01", location="Herrenhausen", filename="test_year.nc")

"${python_root}/miniforge3/envs/Stadtwetter/bin/python3" "${path_root}/auto_convert.py" $path_input $path_output $debug
echo "Converting finished"

#cd $root_root
path_input_forecast="/data/stadtwetter/netcdf_daten/"
path_output_forecast="/data/stadtwetter/Vorhersage"

#
"${python_root}/miniforge3/envs/Stadtwetter/bin/python3" "${path_root}/auto_forecast.py" $path_input_forecast $path_output_forecast $debug
echo " Forecast finished"



path_input_measured="/data/stadtwetter/netcdf_daten/latest_herrenhausen_res_imuknet1.nc"
path_input_forecast_single="/data/stadtwetter/Vorhersage/forecast_test_single_multiday.nc"
path_input_forecast_multi="/data/stadtwetter/Vorhersage/forecast_test_multi_multiday.nc"

path_output="/data/stadtwetter/Vorhersage/Grafiken/test/"
path_output="/home/stadtwetter/public_html/jsons/"

debug=1  # true

"${python_root}/miniforge3/envs/Stadtwetter/bin/python3" "${path_root}/auto_visualisation.py" $path_input_measured $path_input_forecast_single $path_input_forecast_multi $path_output $debug
echo " Visuals finished"


path_ongoing="/data/stadtwetter/Vorhersage/model_data_for_evaluation.nc"

"${python_root}/miniforge3/envs/Stadtwetter/bin/python3" "${path_root}/auto_archive.py"  $path_input_forecast_single $path_input_forecast_multi $path_ongoing $debug #$path_template