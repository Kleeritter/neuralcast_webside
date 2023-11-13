   . ~/.bashrc_miniconda3
conda activate stadtwetter
#export PYTHONPATH=$PYTHONPATH:/localdata/weathermaps/imuk

path_input=/data/datenarchiv/imuk/
year=2022
full="False"
startday="2007-01-01"
endday="2007-12-31"
location="Herrenhausen"
#filename="test_year.nc"
filename="/data/stadtwetter/netcdf_daten/archiv/2007.nc"
#path_output=/localdata/weathermaps/webside/gross



#convert_years(path="/data/datenarchiv/imuk/", year=2022,full=True, startday="2022-01-01",endday="2022-03-01", location="Herrenhausen", filename="test_year.nc")

python /home/stadtwetter/neuralcast_webside/converting/manual_convert.py $path_input $year $full $startday $endday $location $filename
echo "Finished"