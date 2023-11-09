   . ~/.bashrc_miniconda3
conda activate stadtwetter
#export PYTHONPATH=$PYTHONPATH:/localdata/weathermaps/imuk

path_input=/data/datenarchiv/imuk/
year= 2022
full= "True"
startday= "2022-01-01"
endday= "2022-03-01"
location="Herrenhausen"
filename="test_year.nc"
#path_output=/localdata/weathermaps/webside/gross



#convert_years(path="/data/datenarchiv/imuk/", year=2022,full=True, startday="2022-01-01",endday="2022-03-01", location="Herrenhausen", filename="test_year.nc")

python /home/stadtwetter/neuralcast_webside/converting/auto_convert.py $path_input $year $full $startday $endday $location $filename
echo "Finished"