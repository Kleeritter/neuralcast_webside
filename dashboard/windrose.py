import plotly.express as px
import xarray as xr
import numpy as np 
import pandas as pd

def prepare_data_for_windrose(df):
#ds = xr.open_dataset("testdata/stadtwetter/netcdf_daten/2024/02/2024-02_herrenhausen.nc")
# Incorporate data
#df = ds.to_dataframe()
#print(df[["sonic_Wind_Dir", "sonic_Wind_Speed"]])


    #direction_bins = np.arange(-11.25, 371.25, 22.5)  # 17 Kanten für 16 Bins
    direction_bins = np.linspace(0,360,9)
    direction_labels = ['N',  'NE', 'E', 'SE', 'S', 'SW',   'W',  'NW',]

    #direction_labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    df['wind_dir_bin'] = pd.cut(df['sonic_Wind_Dir'], bins=direction_bins, labels=direction_labels, ordered=False)

    # Bin-Einteilung für die Windgeschwindigkeit (z.B. in 5 m/s Schritten)
    speed_bins = [0, 5, 8, 10, 20]#[0, 1, 3, 6, 10, 15, 20]  # 7 Kanten für 6 Bins
    speed_labels = ['<5 m/s', '5-8 m/s', '8-10 m/s', '>10 m/s']#['0-1 m/s', '1-3 m/s', '3-6 m/s', '6-10 m/s', '10-15 m/s', '>15 m/s']
    df['wind_speed_bin'] = pd.cut(df['sonic_Wind_Speed'], bins=speed_bins, labels=speed_labels, include_lowest=True)

    # Gruppieren der Daten
    windrose_data = df.groupby(['wind_dir_bin', 'wind_speed_bin']).size().reset_index(name='frequency')


   # print(windrose_data)
    return windrose_data