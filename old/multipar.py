import seaborn as sns
import matplotlib.pyplot as plt
from funcs.visualer_funcs import lstm_uni, multilstm_full, start_index_test,start_index_real,end_index_test,end_index_real, conv
from datetime import datetime
import pandas as pd
import itertools
from funcs.trad.sarima import sarima
from funcs.trad.p_ro import pp
# Passe den Dateipfad entsprechend an

forecastvar="temp"

datetime_str = '07/16/22 00:00:00'

datetime_object = datetime.strptime(datetime_str, '%m/%d/%y %H:%M:%S')
# Passe den Dateipfad entsprechend an
nc_path = '../Data/stunden/2022_resample_stunden.nc'
import xarray as xr

data = xr.open_dataset(nc_path)
wintertag = pd.to_datetime("2022-12-16 00:00")  # Datum im datetime64-Format
sommmertag=pd.to_datetime("2022-07-16 00:00")
frühlingstag = pd.to_datetime("2022-04-16 00:00")  # Datum im datetime64-Format
herbsttag=pd.to_datetime("2022-10-16 00:00")

#print(dataf.index.get_loc(gesuchtes_datum))
# Indexbereich für den 16. Juli 2011 ermitteln


# Den von Ihnen gesuchten Indexbereich erhalten Sie mit:
#print(start_index["index"])
#print(data["index"])
#print(data["index"][np.datetime64('2022-07-16T00:00:00')])#["2022-01-01 00:00"])
#time_real= data["index"][start_index_real-1:end_index_real]
time_prog=data["index"]#[start_index_real:end_index_real]
#real_values_sommer = data['temp'][start_index_real-1:end_index_real].values.reshape((25,))
singledata_sommer= data['temp']#[start_index_test:end_index_test].values
data_full=xr.open_dataset(nc_path)[['temp',"press_sl","humid","Geneigt CM-11","gust_10","gust_50", "rain", "wind_10", "wind_50"]]
data_fuller=xr.open_dataset(nc_path)[['temp',"press_sl","humid","Geneigt CM-11","gust_10","gust_50", "rain", "wind_10", "wind_50","wind_dir_50"]]
data_ligth=xr.open_dataset(nc_path)[['temp',"press_sl","humid"]]
##predicted_values.append(predicted_value.item())
# Passe die Achsenbezeichnungen und das Layout entsprechend an
sns.set_theme(style="darkgrid")

Messsomer =data['temp'][start_index_real(nc_path,sommmertag):end_index_real(nc_path,sommmertag)].values.reshape((48,))
Messwinter =data['temp'][start_index_real(nc_path,wintertag):end_index_real(nc_path,wintertag)].values.reshape((48,))
Messfrühling=data['temp'][start_index_real(nc_path,frühlingstag):end_index_real(nc_path,frühlingstag)].values.reshape((48,))
Messherbst=  data['temp'][start_index_real(nc_path,herbsttag):end_index_real(nc_path,herbsttag)].values.reshape((48,))

sommer=pd.DataFrame({
    'Datum': time_prog[start_index_real(nc_path,sommmertag):end_index_real(nc_path,sommmertag)],#pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'Messdaten' : Messsomer,
    'Univariantes LSTM': list(itertools.chain(Messsomer[0:24],lstm_uni("output/lstm_model_frisch.pth",singledata_sommer,start_index=start_index_test(nc_path,sommmertag),end_index=end_index_test(nc_path,sommmertag)))),#.insert(0, Messsomer[0]),
    #'Multivariantes LSTM_3': list(itertools.chain(Messsomer[0:24],multilstm_light("output/lstm_model_multi_3var.pth",data_ligth,start_idx=start_index_test(nc_path,sommmertag),end_idx=end_index_test(nc_path,sommmertag)))),#.insert(0, Messsomer[0:24]),
    'Multivariantes LSTM': list(itertools.chain(Messsomer[0:24],multilstm_full("output/lstm_model_multi_17var.pth",data_fuller,start_idx=start_index_test(nc_path,sommmertag),end_idx=end_index_test(nc_path,sommmertag)))),#.insert(0, Messsomer[0:24])
    'SARIMA' : list(itertools.chain(Messsomer[0:24],sarima(nc_path,sommmertag))),
    'PROPHET' : list(itertools.chain(Messsomer[0:24],pp(nc_path,sommmertag))),
    'CONV':list(itertools.chain(Messsomer[0:24],conv("output/lstm_model_conv.pth",data_fuller,start_idx=start_index_test(nc_path,sommmertag),end_idx=end_index_test(nc_path,sommmertag))))
})

winter=pd.DataFrame({
    'Datum': time_prog[start_index_real(nc_path,wintertag):end_index_real(nc_path,wintertag)],#pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'Messdaten' : Messwinter,
    'Univariantes LSTM': list(itertools.chain(Messwinter[0:24],lstm_uni("output/lstm_model_frisch.pth",singledata_sommer,start_index=start_index_test(nc_path,wintertag),end_index=end_index_test(nc_path,wintertag)))),#.insert(0, Messwinter[0:24]),
    #'Multivariantes LSTM_3':list(itertools.chain(Messwinter[0:24],multilstm_light("output/lstm_model_multi_3var.pth",data_ligth,start_idx=start_index_test(nc_path,wintertag),end_idx=end_index_test(nc_path,wintertag)))),#.insert(0, Messwinter[0:24]),
    'Multivariantes LSTM':list(itertools.chain(Messwinter[0:24],multilstm_full("output/lstm_model_multi_17var.pth",data_fuller,start_idx=start_index_test(nc_path,wintertag),end_idx=end_index_test(nc_path,wintertag)))),#.insert(0, Messwinter[0:24])
    'SARIMA' : list(itertools.chain(Messwinter[0:24],sarima(nc_path,wintertag))),
    'PROPHET' : list(itertools.chain(Messwinter[0:24],pp(nc_path,wintertag))),
    'CONV':list(itertools.chain(Messwinter[0:24],conv("output/lstm_model_conv.pth",data_fuller,start_idx=start_index_test(nc_path,wintertag),end_idx=end_index_test(nc_path,wintertag))))
})

herbst=pd.DataFrame({
    'Datum': time_prog[start_index_real(nc_path,herbsttag):end_index_real(nc_path,herbsttag)],#pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'Messdaten' :Messherbst,
    'Univariantes LSTM': list(itertools.chain(Messherbst[0:24],lstm_uni("output/lstm_model_frisch.pth",singledata_sommer,start_index=start_index_test(nc_path,herbsttag),end_index=end_index_test(nc_path,herbsttag)))),#.insert(0, Messherbst[0:24]),
    #'Multivariantes LSTM_3':list(itertools.chain(Messherbst[0:24],multilstm_light("output/lstm_model_multi_3var.pth",data_ligth,start_idx=start_index_test(nc_path,herbsttag),end_idx=end_index_test(nc_path,herbsttag)))),#.insert(0, Messherbst[0:24]),
    'Multivariantes LSTM': list(itertools.chain(Messherbst[0:24],multilstm_full("output/lstm_model_multi_17var.pth",data_fuller,start_idx=start_index_test(nc_path,herbsttag),end_idx=end_index_test(nc_path,herbsttag)))),
    'SARIMA' : list(itertools.chain(Messherbst[0:24],sarima(nc_path,herbsttag))),
    'PROPHET' : list(itertools.chain(Messherbst[0:24],pp(nc_path,herbsttag))),
    'CONV':list(itertools.chain(Messherbst[0:24],conv("output/lstm_model_conv.pth",data_fuller,start_idx=start_index_test(nc_path,herbsttag),end_idx=end_index_test(nc_path,herbsttag))))
})

frühling=pd.DataFrame({
    'Datum': time_prog[start_index_real(nc_path,frühlingstag):end_index_real(nc_path,frühlingstag)],#pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'Messdaten' :Messfrühling ,
    'Univariantes LSTM': list(itertools.chain(Messfrühling[0:24],lstm_uni("output/lstm_model_frisch.pth",singledata_sommer,start_index=start_index_test(nc_path,frühlingstag),end_index=end_index_test(nc_path,frühlingstag)))),#.insert(0, Messfrühling[0:24]),
    #'Multivariantes LSTM_3':list(itertools.chain(Messfrühling[0:24],multilstm_light("output/lstm_model_multi_3var.pth",data_ligth,start_idx=start_index_test(nc_path,frühlingstag),end_idx=end_index_test(nc_path,frühlingstag)))),#.insert(0, Messfrühling[0:24]),
    'Multivariantes LSTM': list(itertools.chain(Messfrühling[0:24],multilstm_full("output/lstm_model_multi_17var.pth",data_fuller,start_idx=start_index_test(nc_path,frühlingstag),end_idx=end_index_test(nc_path,frühlingstag)))),
    'SARIMA' : list(itertools.chain(Messfrühling[0:24],sarima(nc_path,frühlingstag))),
    'PROPHET' : list(itertools.chain(Messfrühling[0:24],pp(nc_path,frühlingstag))),
    'CONV':list(itertools.chain(Messfrühling[0:24],conv("output/lstm_model_conv.pth",data_fuller,start_idx=start_index_test(nc_path,frühlingstag),end_idx=end_index_test(nc_path,frühlingstag))))
})

fig, axs= plt.subplots(2,2,figsize=(12,8))
sns.set(style="darkgrid")  # Setze den Stil des Plots auf "darkgrid"
#sns.scatterplot(x='Datum', y='Univariantes LSTM', data=frühling, ax=axs[0, 0])
sns.lineplot(x='Datum', y='value', hue='variable', data=pd.melt(frühling, ['Datum']),markers=True, style="variable", ax=axs[0,0])
sns.lineplot(x='Datum', y='value', hue='variable', data=pd.melt(sommer, ['Datum']),markers=True, style="variable",ax=axs[0,1])
sns.lineplot(x='Datum', y='value', hue='variable', data=pd.melt(herbst, ['Datum']),markers=True, style="variable", ax=axs[1,0])
sns.lineplot(x='Datum', y='value', hue='variable', data=pd.melt(winter, ['Datum']),markers=True, style="variable", ax=axs[1,1])

# Diagramm anzeigen
plt.tight_layout()
plt.show()

sns.lineplot(x='Datum', y='value', hue='variable', data=pd.melt(herbst, ['Datum']),markers=True, style="variable")
plt.show()