import plotly.express as px
import xarray as xr
import pandas as pd
from datetime import timedelta

forecast_var="derived_Press_sl"
dataset = xr.open_dataset("latest_herrenhausen_res_imuknet1.nc")
dataset_forecast_multi = xr.open_dataset("forecast_test.nc")
dataset_forecast_single = xr.open_dataset("forecast_test_single.nc")
#print(dataset.head())
df =dataset.to_dataframe()
# Den Index um eine Stunde verschieben
new_index = df.index + pd.DateOffset(hours=1)

# Das DataFrame mit dem verschobenen Index erstellen
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

fig = px.line(merged_df, title='Life expectancy in Canada')


fig.write_html("test.html")