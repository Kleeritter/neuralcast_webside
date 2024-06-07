import plotly.express as px
import xarray as xr
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import mean_squared_error#,root_mean_squared_error
from pytz import timezone
import yaml
import os
import json
from jinja2 import Template

def visualize_var(forecast_var="derived_Press_sl", measured_data_path="latest_herrenhausen_res_imuknet1.nc",forecast_multi_path="forecast_test.nc",forecast_single_path="forecast_test_single.nc", outputpath="",debug=False):
    #forecast_var="derived_Press_sl"
    dataset = xr.open_dataset(measured_data_path)
    dataset_forecast_single = xr.open_dataset(forecast_single_path)

    dataset_multi = xr.open_dataset(forecast_multi_path)

       # Finde alle Variablen im Dataset, deren Name "Temperatur" enthält
    temperature_vars = [var for var in dataset.variables if 'Temperatur' in var]
    press_vars = [var for var in dataset.variables if 'Press_sl' in var]
    humid_vars = [var for var in dataset.variables if 'Feuchte' in var]


    dataset = rescale_var(dataset=dataset)
    dataset_forecast_single = rescale_var(dataset=dataset_forecast_single)
    dataset_multi = rescale_var(dataset=dataset_multi)

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
    best_model = attribute_data[forecast_var]["best_model"]

    print(unit)

    ### Measured Values

    df = dataset.to_dataframe()

    df= df[forecast_var][-48:]

    ### Single Forecast



    last_forecast_hour = dataset_forecast_single.attrs["last_forecast_hour"]
    df_single =dataset_forecast_single.to_dataframe()


    df_single = df_single[forecast_var+"_"+last_forecast_hour][-25:] 

    forecasts_single = dataset_forecast_single.to_dataframe().filter(regex=f'^{forecast_var}')[-25:]

    formins =forecasts_single.min(axis=1).values

    formaxs =forecasts_single.max(axis=1).values

    selected_columns = dataset_forecast_single.to_dataframe().filter(regex=f'^{forecast_var}')[-72:-24] # Remove  [:-24] for multi line  version



    #selected_columns.iloc[(-24+ int(last_forecast_hour)):, selected_columns.columns.get_loc(forecast_var+"_"+last_forecast_hour)] = pd.NA
    df_single_old = selected_columns


    # Minimum- und Maximumwerte für jeden Index extrahieren
    mins = df_single_old.min(axis=1).values
    maxs = df_single_old.max(axis=1).values
    means = df_single_old.mean(axis=1).values


    #### Multi Forecast
    last_forecast_hour_multi = dataset_multi.attrs["last_forecast_hour"]
    df_multi =dataset_multi.to_dataframe()

    forecasts_m = dataset_multi.to_dataframe().filter(regex=f'^{forecast_var}')[-25:]

    forminm =forecasts_m.min(axis=1).values

    formaxm =forecasts_m.max(axis=1).values

    df_multi = df_multi[forecast_var+"_"+last_forecast_hour_multi][-25:] 
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
    print("Means:", merged_df['Messwerte'][abs(48-len(means)):47])
    #### Single

    #rms_single = root_mean_squared_error(merged_df['Messwerte'][abs(47-len(means)):47], means)
    rms_single = mean_squared_error(merged_df['Messwerte'][abs(48-len(means)):48], means,squared=False)

    #### Multi


    #rms_multi = root_mean_squared_error(merged_df['Messwerte'][abs(47-len(means_multi)):47], means_multi)
    rms_multi = mean_squared_error(merged_df['Messwerte'][abs(48-len(means_multi)):48], means_multi,squared=False)



    merged_df.loc[merged_df.index[-25], "ImuKnet Single"] = merged_df.loc[merged_df.index[-25], "Messwerte"]
    merged_df.loc[merged_df.index[-25], "ImuKnet Multi"] = merged_df.loc[merged_df.index[-25], "Messwerte"]
    print(merged_df)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['Messwerte'], mode='lines+markers', name='Messwerte',line=dict(color="#000000")))
    if best_model == "single":

        #attribute_data[forecast_var]['RMSE'] = rms_single
        try:
            with open("neuralcast_webside/visuals/units.yml", 'r') as yaml_file:
                data = yaml.safe_load(yaml_file)
        except FileNotFoundError:
            # Wenn die Datei nicht existiert, erstellen Sie ein leeres Datenobjekt
            data = {}
        
        data.setdefault(forecast_var, {})['RMSE'] = str(rms_single)

        # Schreiben Sie die aktualisierten Daten zurück in die YAML-Datei
        print(debug)
        if not debug:
            with open("/home/stadtwetter/public_html/units.json", 'w') as json_file:
                json.dump(data, json_file)

        fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df["ImuKnet Single"],mode='lines+markers', name="ML Vorhersage",line=dict(color="#FF0000")))

        #fig.add_trace(go.Scatter(x=df_single_old.index, y=means, mode='lines+markers', name="Mittelwert der Vorhersagen",legendgroup="group1",  legendgrouptitle_text="Vorherige Vorhersagen Univariant"))


       # Schatten der Zukunft:
        fig.add_trace(  go.Scatter(  visible="legendonly",      name='Unsicherheit der Vorhersagen',        x=forecasts_single.index,        y=formaxs,        mode='lines',        marker=dict(color="#444"),        line=dict(width=0),        showlegend=True,  legendgroup="group3",legendgrouptitle_text="Unsicherheit der Vorhersagen" ))

        fig.add_trace(go.Scatter(     visible="legendonly",    name='Lower Bound',        x=forecasts_single.index,        y=formins,        marker=dict(color="#444"),        line=dict(width=0),        mode='lines',        fillcolor='rgba(255, 0, 0, 0.5)',        fill='tonexty',        showlegend=False,   legendgroup="group3" ))
    
        #Scjatten der Vergangenheit
        fig.add_trace(  go.Scatter(   visible="legendonly",     name='Vorherige Vorhersagen',        x=merged_df.index,        y=maxs,        mode='lines',        marker=dict(color="#444"),        line=dict(width=0),        showlegend=True  ,legendgroup="group2", legendgrouptitle_text="Vorherige Vorhersagen"  ))

        fig.add_trace(go.Scatter(   visible="legendonly",     name='Lower Bound',        x=merged_df.index,        y=mins,        marker=dict(color="#444"),        line=dict(width=0),        mode='lines',        fillcolor='rgba(255, 0, 0, 0.5)',        fill='tonexty',        showlegend=False   ,legendgroup="group2" ))


        rmse_string = str(round(rms_single,2))+unit

    elif best_model == "multi":

        try:
            with open("neuralcast_webside/visuals/units.yml", 'r') as yaml_file:
                data = yaml.safe_load(yaml_file)
        except FileNotFoundError:
            # Wenn die Datei nicht existiert, erstellen Sie ein leeres Datenobjekt
            data = {}
        
        data.setdefault(forecast_var, {})['RMSE'] = str(rms_multi)

        # Schreiben Sie die aktualisierten Daten zurück in die YAML-Datei
        #with open("neuralcast_webside/visuals/units.json", 'w') as json_file:
         #   json.dump(data, json_file)
        if not debug:
            with open("/home/stadtwetter/public_html/units.json", 'w') as json_file:
                json.dump(data, json_file)

        fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df["ImuKnet Multi"],mode='lines+markers', name="ML Vorhersage",line=dict(color="#FF0000")))
        #fig.add_trace(go.Scatter(x=df_multi_old.index, y=means_multi, mode='lines+markers', name="Mittelwert der Vorhersagen",legendgroup="group2",  legendgrouptitle_text="Vorherige Vorhersagen Multivariant"))

       # Schatten der Zukunft:
        fig.add_trace(  go.Scatter(  visible="legendonly",      name='Unsicherheit der Vorhersagen',        x=forecasts_m.index,        y=formaxm,        mode='lines',        marker=dict(color="#444"),        line=dict(width=0),        showlegend=True,  legendgroup="group3",legendgrouptitle_text="Unsicherheit der Vorhersagen" ))

        fig.add_trace(go.Scatter(     visible="legendonly",    name='Lower Bound',        x=forecasts_m.index,        y=forminm,        marker=dict(color="#444"),        line=dict(width=0),        mode='lines',        fillcolor='rgba(255, 0, 0, 0.5)',        fill='tonexty',        showlegend=False,   legendgroup="group3" ))
    
       
       # Schatten der Vergangenheit
       
        fig.add_trace(  go.Scatter(  visible="legendonly",      name='Vorherige Vorhersagen',        x=df_multi_old.index,        y=maxs_multi,        mode='lines',        marker=dict(color="#444"),        line=dict(width=0),        showlegend=True,  legendgroup="group2",legendgrouptitle_text="Vorherige Vorhersagen" ))

        fig.add_trace(go.Scatter(     visible="legendonly",    name='Lower Bound',        x=df_multi_old.index,        y=mins_multi,        marker=dict(color="#444"),        line=dict(width=0),        mode='lines',        fillcolor='rgba(255, 0, 0, 0.5)',        fill='tonexty',        showlegend=False,   legendgroup="group2" ))
    
        rmse_string =  str(round(rms_multi,2))+unit

    else:
        fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df["ImuKnet Single"],mode='lines+markers', name="ImuKnet Single"))
        fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df["ImuKnet Multi"],mode='lines+markers', name="ImuKnet Multi"))
        rmse_string = "Single= "+str(round(rms_single,2))+unit+"Multi= "+ str(round(rms_multi,2))+unit 


    fig.update_layout(showlegend=True)
    


    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    fig.update_layout( xaxis_title='Zeit (MESZ)',                  yaxis_title=longname  +" ("+unit+")")

    if debug:  fig.write_html(outputpath +forecast_var + ".html")
    fig.write_json(file=outputpath+forecast_var+".json")
    
    return


def rescale_var(dataset):
    #Function to rescale Variables for public, e.g temperature in °C

    temperature_vars = [var for var in dataset.variables if 'Temperatur' in var]
    press_vars = [var for var in dataset.variables if 'Press_sl' in var]
    humid_vars = [var for var in dataset.variables if 'Feuchte' in var]

    for var_name in temperature_vars:
        dataset[var_name] -= 273.15

    for var_name in press_vars:
        dataset[var_name] /= 100

    for var_name in press_vars:
        dataset[var_name] = np.clip(dataset[var_name] * 100, None, 100)


    return dataset