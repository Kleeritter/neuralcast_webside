source /home/stadtwetter/.bashrc
#conda init bash
/home/stadtwetter/miniforge3/bin/conda activate Stadtwetter
#export PYTHONPATH=$PYTHONPATH:/localdata/weathermaps/imuk

path_input=/data/stadtwetter/netcdf_daten/
path_output=/data/stadtwetter/Vorhersage/

/home/stadtwetter/miniforge3/envs/Stadtwetter/bin/python3 /home/stadtwetter/neuralcast_webside/auto_forecast.py $path_input $path_output
echo "Finished"