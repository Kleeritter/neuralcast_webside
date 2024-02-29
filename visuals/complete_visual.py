import plotly.express as px
import xarray as xr
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go


def visualize_var(forecast_var="derived_Press_sl", measured_data_path="latest_herrenhausen_res_imuknet1.nc",forecast_multi_path="forecast_test.nc",forecast_single_path="forecast_test_single.nc", outputpath=""):
    #forecast_var="derived_Press_sl"
    dataset = xr.open_dataset(measured_data_path)
    #dataset_forecast_multi = xr.open_dataset(forecast_multi_path)
    dataset_forecast_single = xr.open_dataset(forecast_single_path)
    #print(dataset.head())
    df = dataset.to_dataframe()
    # Den Index um eine Stunde verschieben
    #new_index = df.index + pd.DateOffset(hours=1)

    # DataFrame mit dem verschobenen Index erstellen
    #df = df.set_index(new_index)[forecast_var]
    #print(df)
    df= df[-48:]
    #df_multi =dataset_forecast_multi.to_dataframe()[forecast_var]
    #print(dataset[forecast_var].attrs)
    #print(dataset_forecast_single.attrs)
    #print(dict(dataset[forecast_var].attrs)["units"])
    last_forecast_hour = dataset_forecast_single.attrs["last_forecast_hour"]
    df_single =dataset_forecast_single.to_dataframe()



    df_single = df_single[forecast_var+"_"+last_forecast_hour][-24:]
    selected_columns = dataset_forecast_single.to_dataframe().filter(regex=f'^{forecast_var}')#[:-24]
    print(selected_columns)
    
    #selected_columns.loc[len(selected_columns) - 24:, forecast_var+"_"+last_forecast_hour] = pd.NA
    selected_columns.iloc[(-24+ int(last_forecast_hour)):, selected_columns.columns.get_loc(forecast_var+"_"+last_forecast_hour)] = pd.NA
    #selected_columns = my_series[my_series.index.str.startswith(prefix)]
    df_single_old = selected_columns#dataset_forecast_single.to_dataframe()[selected_columns][:-24]
    #print(df_single)
    #    print(df_multi)
    merged_df = pd.merge(df, df_single, on='time',how='outer', suffixes=('_a', '_b'))
    #merged_df = pd.merge(merged_df, df_multi, on='time',how='outer')
    merged_df=merged_df.rename(columns={forecast_var: "Messwerte", forecast_var+"_"+last_forecast_hour: "ImuKnet Single"})#,forecast_var: "ImuKnet Multi"})
    merged_df = merged_df.join(df_single_old)
    #print(merged_df)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['Messwerte'], mode='lines+markers', name='Messwerte'))
    #fig = px.line(merged_df, x=merged_df.index, y='Messwerte', title=f'{forecast_var}')
    fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df["ImuKnet Single"],
                             mode='lines+markers', name="ImuKnet Single"))


    for col in merged_df.columns.difference([ 'Messwerte', 'ImuKnet Single']):
        fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df[col],
                             mode='lines+markers', name='Vorhersage zu t='+col[-2:]+"h",    legendgroup="group",  # this can be any string, not just "group"
    legendgrouptitle_text="Vorherige Vorhersagen",line=dict(color='rgba(169,169,169,0.25)')))#, line=dict(opacity=0.5)))

    fig.update_layout(showlegend=True)
    fig.update_layout(title='Messwerte und Vorhersagen für '+ dataset[forecast_var].attrs['standard_name'],  xaxis_title='Zeit',
                  yaxis_title=dataset[forecast_var].attrs['longname'] +"["+dataset[forecast_var].attrs["units"]+"]")


    #fig = px.line(merged_df, title= forecast_var)

    fig.write_html(outputpath +forecast_var + ".html")
    #fig.close()
    return