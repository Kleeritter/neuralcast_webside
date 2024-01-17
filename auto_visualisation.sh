source /home/stadtwetter/.bashrc
#conda init bash
/home/stadtwetter/miniforge3/bin/conda activate Stadtwetter
#export PYTHONPATH=$PYTHONPATH:/localdata/weathermaps/imuk

path_input_measured=/data/stadtwetter/netcdf_daten/latest_herrenhausen_res_imuknet1.nc
path_input_forecast_single=/data/stadtwetter/Vorhersage/forecast_multi.nc
path_input_forecast_multi=/data/stadtwetter/Vorhersage/forecast_test.nc
path_output=/data/stadtwetter/Vorhersage/Grafiken

/home/stadtwetter/miniforge3/envs/Stadtwetter/bin/python3 /home/stadtwetter/neuralcast_webside/auto_visualisation.py $path_input_measured $path_input_forecast_single $path_input_forecast_multi $path_output
echo "Finished"