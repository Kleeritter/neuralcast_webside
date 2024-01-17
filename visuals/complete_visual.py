import plotly.express as px
import xarray as xr
import pandas as pd
from datetime import timedelta


def visualize_var(forecast_var="derived_Press_sl", measured_data_path="latest_herrenhausen_res_imuknet1.nc",forecast_multi_path="forecast_test.nc",forecast_single_path="forecast_test_single.nc", outputpath=""):
    #forecast_var="derived_Press_sl"
    dataset = xr.open_dataset(measured_data_path)
    dataset_forecast_multi = xr.open_dataset(forecast_multi_path)
    dataset_forecast_single = xr.open_dataset(forecast_single_path)
    #print(dataset.head())
    df = dataset.to_dataframe()
    # Den Index um eine Stunde verschieben
    new_index = df.index + pd.DateOffset(hours=1)

    # DataFrame mit dem verschobenen Index erstellen
    df = df.set_index(new_index)[forecast_var]
    df_multi =dataset_forecast_multi.to_dataframe()[forecast_var]
    df_single =dataset_forecast_single.to_dataframe()[forecast_var]

    print(df_multi)
    merged_df = pd.merge(df, df_single, on='time',how='outer', suffixes=('_a', '_b'))
    merged_df = pd.merge(merged_df, df_multi, on='time',how='outer')
    merged_df=merged_df.rename(columns={forecast_var+"_a": "Messwerte", forecast_var+"_b": "ImuKnet Single",forecast_var: "ImuKnet Multi"})
    print(merged_df)
    merged_df.at[merged_df.index.max()- timedelta(hours=24),"ImuKnet Single"] = merged_df.at[merged_df.index.max()- timedelta(hours=24),"Messwerte"]
    merged_df.at[merged_df.index.max()- timedelta(hours=24),"ImuKnet Multi"] = merged_df.at[merged_df.index.max()- timedelta(hours=24),"Messwerte"]
    print(merged_df.at[merged_df.index.max()- timedelta(hours=24),"ImuKnet Single"])

    fig = px.line(merged_df, title= forecast_var)


    fig.write_html(outputpath + forecast_var + ".html")

return