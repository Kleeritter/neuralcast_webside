# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import xarray as xr
import plotly.express as px
import dash_bootstrap_components as dbc
from datetime import date

ds = xr.open_dataset("testdata/stadtwetter/netcdf_daten/latest_herrenhausen.nc")
# Incorporate data
df = ds.to_dataframe()


# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# App layout

app.layout = dbc.Container([
    dbc.Row([
        html.Div('My First App with Data, Graph, and Controls', className="text-primary text-center fs-3")
    ]),



    dbc.Row([

        dbc.Col([
            dbc.Row([
            
            dcc.Dropdown(["herrenhausen_Temperatur","herrenhausen_Feuchte", "dach_Diffus_CMP-11","dach_Global_CMP-11","herrenhausen_Gust_Speed",
     "sonic_Gust_Speed","herrenhausen_Regen", "herrenhausen_Wind_Speed", "sonic_Wind_Speed", "sonic_Wind_Dir"
], 'herrenhausen_Temperatur', multi=True,
                         id='radio-buttons-final'),
                         
                         
            dcc.DatePickerRange(
                            id='date1',
                            min_date_allowed=date(2024, 3, 15),
                            max_date_allowed=date(2024, 3, 19),
                            initial_visible_month=date(2024, 3, 15),
                            end_date=date(2024, 3, 19),
                            #with_portal=True
                        ),

            ]),
             dbc.Row([
            dcc.Graph(figure={}, id='my-first-graph-final'),
           
            ]),

            dbc.Row([
            #dash_table.DataTable(data=df.to_dict('records'), page_size=12, style_table={'overflowX': 'auto'})
             dash_table.DataTable(id='table1', page_size=12, style_table={'overflowX': 'auto'}),
             html.Button("Download Excel", id="btn_xlsx"),
            dcc.Download(id="download-dataframe-xlsx"),


            ] ),

  
             
             
            #html.Button("Download Excel", id="btn_xlsx"),
            #dcc.Download(id="download-dataframe-xlsx"),
            #html.Button("Download CSV", id="btn_csv"),
            #dcc.Download(id="download-dataframe-csv"),
             
             
             
             ],width=12),



        ])
        

], fluid=True) 

# Add controls to build the interaction
@callback(
    Output(component_id='my-first-graph-final', component_property='figure'),
    Output(component_id='table1', component_property='columns'),
    Output(component_id='table1', component_property='data'),




    Input(component_id='radio-buttons-final', component_property='value'),

    Input('date1', 'start_date'),
    Input('date1', 'end_date'),

)
def update_graph(col_chosen_1,date1_start,date2_end):
    if not isinstance(col_chosen_1, list):
        col_chosen_1 = [col_chosen_1]
    
    filtered_df = df.loc[date1_start:date2_end]
    print(filtered_df)
    #filtered_df["Date"] =filtered_df.index
    #filtered_df.insert("Date", filtered_df.index, True)
    filtered_df = filtered_df.assign(Date=filtered_df.index)

    fig1 = px.line(filtered_df, x=filtered_df.index, y=col_chosen_1)

    #table1_columns = [{"name": "Date", "id": "Date"}]
    #table1_columns.append({"name": col, "id": col} for col in col_chosen_1)
    table1_columns= [{"name": col, "id": col} for col in col_chosen_1]
    #table1_columns.append({"name": col, "id": col} for col in col_chosen_1)
    table1_columns.insert(0, {"name": "Date", "id": "Date"})  # Fügt die Zahl 2 am Anfang der Liste ein

    table1_data = filtered_df[["Date"]+col_chosen_1].reset_index().to_dict('records')



    return fig1,table1_columns, table1_data#,excel

@callback(
    Output("download-dataframe-xlsx", "data"),
    Input(component_id='radio-buttons-final', component_property='value'),

    Input('date1', 'start_date'),
    Input('date1', 'end_date'),
    Input("btn_xlsx", "n_clicks"),
    prevent_initial_call=True,
)

def excel(col_chosen_1,date1_start,date2_end,nclicks):
    if not isinstance(col_chosen_1, list):
        col_chosen_1 = [col_chosen_1]
    
    filtered_df = df.loc[date1_start:date2_end]
    print(filtered_df)
    #filtered_df["Date"] =filtered_df.index
    #filtered_df.insert("Date", filtered_df.index, True)
    filtered_df = filtered_df.assign(Date=filtered_df.index)
    filtered_df = filtered_df[col_chosen_1]


    excel = dcc.send_data_frame(filtered_df.to_excel, "mydf.xlsx", sheet_name="Sheet_name_1")
    return excel

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
