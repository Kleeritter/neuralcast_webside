import plotly.express as px
import xarray as xr
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go


def visualize_var(forecast_var="derived_Press_sl", measured_data_path="latest_herrenhausen_res_imuknet1.nc",forecast_multi_path="forecast_test.nc",forecast_single_path="forecast_test_single.nc", outputpath=""):
    #forecast_var="derived_Press_sl"
    dataset = xr.open_dataset(measured_data_path)
    dataset_forecast_single = xr.open_dataset(forecast_single_path)

    df = dataset.to_dataframe()

    df= df[forecast_var][-48:]

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
    print("Minimumwerte:", mins)
    print("Maximumwerte:", maxs)
    print("Means:", means)
    merged_df = pd.merge(df, df_single, on='time',how='outer', suffixes=('_a', '_b'))

    merged_df=merged_df.rename(columns={forecast_var: "Messwerte", forecast_var+"_"+last_forecast_hour: "ImuKnet Single"})
    merged_df = merged_df.join(df_single_old)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['Messwerte'], mode='lines+markers', name='Messwerte'))
    fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df["ImuKnet Single"],
                             mode='lines+markers', name="ImuKnet Single"))
    fig.add_trace(go.Scatter(x=df_single_old.index, y=means,
                             mode='lines+markers', name="Vorherige Vorhersagen",color="smoker"))


    fig.add_trace(  go.Scatter(        name='Upper Bound',        x=df_single_old.index,        y=maxs,        mode='lines',        marker=dict(color="#444"),        line=dict(width=0),        showlegend=False    ))

    fig.add_trace(go.Scatter(        name='Lower Bound',        x=df_single_old.index,        y=mins,        marker=dict(color="#444"),        line=dict(width=0),        mode='lines',        fillcolor='rgba(68, 68, 68, 0.3)',        fill='tonexty',        showlegend=False    ))

    #for col in merged_df.columns.difference([ 'Messwerte', 'ImuKnet Single']):
        #fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df[col],mode='lines+markers', name='Vorhersage zu t='+col[-2:]+"h",    legendgroup="group",  legendgrouptitle_text="Vorherige Vorhersagen",line=dict(color='rgba(169,169,169,0.25)')))

    fig.update_layout(showlegend=True)
    fig.update_layout(title='Messwerte und Vorhersagen für '+ dataset[forecast_var].attrs['standard_name'],  xaxis_title='Zeit',
                  yaxis_title=dataset[forecast_var].attrs['longname'] +"["+dataset[forecast_var].attrs["units"]+"]")

    fig.write_html(outputpath +forecast_var + ".html")
    
    return