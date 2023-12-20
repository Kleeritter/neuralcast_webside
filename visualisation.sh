source /home/stadtwetter/.bashrc
#conda init bash
/home/stadtwetter/miniforge3/bin/conda activate Stadtwetter
#export PYTHONPATH=$PYTHONPATH:/localdata/weathermaps/imuk

path_input_measured=/data/stadtwetter/netcdf_daten/
path_input_forecast=/data/stadtwetter/Vorhersage/
path_output=/data/stadtwetter/Vorhersage/grafiken

/home/stadtwetter/miniforge3/envs/Stadtwetter/bin/python3 /home/stadtwetter/neuralcast_webside/auto_visualisation.py $path_input_measured $path_input_forecast $path_output
echo "Finished"