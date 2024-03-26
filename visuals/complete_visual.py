import plotly.express as px
import xarray as xr
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import mean_squared_error,root_mean_squared_error
from pytz import timezone
import yaml
import os

def visualize_var(forecast_var="derived_Press_sl", measured_data_path="latest_herrenhausen_res_imuknet1.nc",forecast_multi_path="forecast_test.nc",forecast_single_path="forecast_test_single.nc", outputpath=""):
    #forecast_var="derived_Press_sl"
    dataset = xr.open_dataset(measured_data_path)
    dataset_forecast_single = xr.open_dataset(forecast_single_path)

    dataset_multi = xr.open_dataset(forecast_multi_path)

       # Finde alle Variablen im Dataset, deren Name "Temperatur" enthält
    temperature_vars = [var for var in dataset.variables if 'Temperatur' in var]
    press_vars = [var for var in dataset.variables if 'Press_sl' in var]

    # Ziehe von den Werten dieser Variablen 278,15 ab
    for var_name in temperature_vars:
        dataset[var_name] -= 278.15
    for var_name in press_vars:
        dataset[var_name] /= 100
    temperature_vars = [var for var in dataset_forecast_single.variables if 'Temperatur' in var]
    press_vars = [var for var in dataset_forecast_single.variables if 'Press_sl' in var]

    # Ziehe von den Werten dieser Variablen 278,15 ab
    for var_name in temperature_vars:
        dataset_forecast_single[var_name] -= 278.15
    for var_name in press_vars:
        dataset_forecast_single[var_name] /= 100

    temperature_vars = [var for var in dataset_multi.variables if 'Temperatur' in var]
    press_vars = [var for var in dataset_multi.variables if 'Press' in var]


    # Ziehe von den Werten dieser Variablen 278,15 ab
    for var_name in temperature_vars:
        dataset_multi[var_name] -= 278.15
    for var_name in press_vars:
        dataset_multi[var_name] /= 100

    ##change time

    mesz_timezone = timezone('Europe/Berlin')
    dataset['time'] = pd.to_datetime(dataset['time'].values).tz_localize('UTC').tz_convert('Europe/Berlin')
    dataset_forecast_single['time'] = pd.to_datetime(dataset_forecast_single['time'].values).tz_localize('UTC').tz_convert('Europe/Berlin')
    dataset_multi['time'] = pd.to_datetime(dataset_multi['time'].values).tz_localize('UTC').tz_convert('Europe/Berlin')

    print(os.getcwd())
    ## select units
    with open("neuralcast_webside/visuals/units.yml", 'r') as yaml_file:
        attribute_data = yaml.safe_load(yaml_file)
    
    unit = attribute_data[forecast_var]["units"]
    longname = attribute_data[forecast_var]["longname"]

    print(unit)

    ### Measured Values

    df = dataset.to_dataframe()

    df= df[forecast_var][-48:]

    ### Single Forecast



    last_forecast_hour = dataset_forecast_single.attrs["last_forecast_hour"]
    df_single =dataset_forecast_single.to_dataframe()


    df_single = df_single[forecast_var+"_"+last_forecast_hour][-24:] 
    selected_columns = dataset_forecast_single.to_dataframe().filter(regex=f'^{forecast_var}')[-72:-24] # Remove  [:-24] for multi line  version



    #selected_columns.iloc[(-24+ int(last_forecast_hour)):, selected_columns.columns.get_loc(forecast_var+"_"+last_forecast_hour)] = pd.NA
    df_single_old = selected_columns


    # Minimum- und Maximumwerte für jeden Index extrahieren
    mins = df_single_old.min(axis=1).values
    maxs = df_single_old.max(axis=1).values
    means = df_single_old.mean(axis=1).values
    #print("Minimumwerte:", mins)
    #print("Maximumwerte:", maxs)
 




    #### Multi Forecast
    last_forecast_hour_multi = dataset_multi.attrs["last_forecast_hour"]
    df_multi =dataset_multi.to_dataframe()


    df_multi = df_multi[forecast_var+"_"+last_forecast_hour_multi][-24:] 
    selected_columns = dataset_multi.to_dataframe().filter(regex=f'^{forecast_var}')[-72:-24]

    df_multi_old = selected_columns

    # Minimum- und Maximumwerte für jeden Index extrahieren
    mins_multi = df_multi_old.min(axis=1).values
    maxs_multi = df_multi_old.max(axis=1).values
    means_multi = df_multi_old.mean(axis=1).values



    merged_df = pd.merge(df, df_single, on='time',how='outer', suffixes=('_a', '_single',))
    print(merged_df)
    merged_df=merged_df.rename(columns={forecast_var: "Messwerte", forecast_var+"_"+last_forecast_hour: "ImuKnet Single"})
    merged_df = merged_df.join(df_single_old)
    merged_df = pd.merge(merged_df, df_multi, on='time',how='outer', suffixes=('', '_multi',))
    merged_df=merged_df.rename(columns={forecast_var+"_"+last_forecast_hour +"_multi": "ImuKnet Multi"})

    ### Calculate RMSEs
    print("Means:", means)
    print("Means:", merged_df['Messwerte'][abs(47-len(means)):47])
    #### Single

    rms_single = root_mean_squared_error(merged_df['Messwerte'][abs(47-len(means)):47], means)
    #### Multi


    rms_multi = root_mean_squared_error(merged_df['Messwerte'][abs(47-len(means_multi)):47], means_multi)
    ### Plotting





  

    print(merged_df)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['Messwerte'], mode='lines+markers', name='Messwerte'))
    fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df["ImuKnet Single"],mode='lines+markers', name="ImuKnet Single"))
    fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df["ImuKnet Multi"],mode='lines+markers', name="ImuKnet Multi"))

    # Vorherige Single                   
    fig.add_trace(go.Scatter(x=df_single_old.index, y=means, mode='lines+markers', name="Mittelwert der Vorhersagen",legendgroup="group1",  legendgrouptitle_text="Vorherige Vorhersagen Univariant"))

    fig.add_trace(  go.Scatter(        name='Upper Bound',        x=df_single_old.index,        y=maxs,        mode='lines',        marker=dict(color="#444"),        line=dict(width=0),        showlegend=False  ,legendgroup="group1"  ))

    fig.add_trace(go.Scatter(        name='Lower Bound',        x=df_single_old.index,        y=mins,        marker=dict(color="#444"),        line=dict(width=0),        mode='lines',        fillcolor='rgba(68, 68, 68, 0.3)',        fill='tonexty',        showlegend=False   ,legendgroup="group1" ))

    # Vorherige Multi 
    fig.add_trace(go.Scatter(x=df_multi_old.index, y=means_multi, mode='lines+markers', name="Mittelwert der Vorhersagen",legendgroup="group2",  legendgrouptitle_text="Vorherige Vorhersagen Multivariant"))

    fig.add_trace(  go.Scatter(        name='Upper Bound',        x=df_multi_old.index,        y=maxs_multi,        mode='lines',        marker=dict(color="#444"),        line=dict(width=0),        showlegend=False,  legendgroup="group2" ))

    fig.add_trace(go.Scatter(        name='Lower Bound',        x=df_multi_old.index,        y=mins_multi,        marker=dict(color="#444"),        line=dict(width=0),        mode='lines',        fillcolor='rgba(68, 68, 68, 0.3)',        fill='tonexty',        showlegend=False,   legendgroup="group2" ))
    fig.add_annotation(
        x=np.max(merged_df.index)-timedelta(hours=5),
        y=max(np.max(merged_df["ImuKnet Single"]),np.max(merged_df["ImuKnet Multi"]))+3,
        xref="x",
        yref="y",
        text="Rmse <br> Single= "+str(round(rms_single,2))+unit+" <br> Multi= "+ str(round(rms_single,2))+unit ,
        showarrow=False,
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
            ),
        align="center",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=0.8
        )
  
    #for col in merged_df.columns.difference([ 'Messwerte', 'ImuKnet Single']):
        #fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df[col],mode='lines+markers', name='Vorhersage zu t='+col[-2:]+"h",    legendgroup="group",  legendgrouptitle_text="Vorherige Vorhersagen",line=dict(color='rgba(169,169,169,0.25)')))

    fig.update_layout(showlegend=True)
    
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    fig.update_layout(title='Messwerte und Vorhersagen für '+ longname,  xaxis_title='Zeit (MESZ)',                  yaxis_title=longname  +"["+unit+"]")

    fig.write_html(outputpath +forecast_var + ".html")
    
    return


