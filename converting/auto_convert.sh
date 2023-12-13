bash /home/stadtwetter/.bashrc
#conda init bash
conda activate Stadtwetter
#export PYTHONPATH=$PYTHONPATH:/localdata/weathermaps/imuk

path_input=/data/datenarchiv/imuk/
path_output=/data/stadtwetter/netcdf_daten/
#path_output=/localdata/weathermaps/webside/gross



#convert_years(path="/data/datenarchiv/imuk/", year=2022,full=True, startday="2022-01-01",endday="2022-03-01", location="Herrenhausen", filename="test_year.nc")

python /home/stadtwetter/neuralcast_webside/converting/auto_convert.py $path_input $path_output
echo "Finished"