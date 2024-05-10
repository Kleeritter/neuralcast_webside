# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import xarray as xr
import plotly.express as px
import dash_bootstrap_components as dbc
from datetime import date


import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

ds = xr.open_dataset("testdata/stadtwetter/netcdf_daten/2024/02/2024-02_herrenhausen.nc")
# Incorporate data
df = ds.to_dataframe()

# App layout
cols =["herrenhausen_Temperatur","herrenhausen_Feuchte", "dach_Diffus_CMP-11","dach_Global_CMP-11","herrenhausen_Gust_Speed",
     "sonic_Gust_Speed","herrenhausen_Regen", "herrenhausen_Wind_Speed", "sonic_Wind_Speed", "sonic_Wind_Dir"
]

app.layout = dbc.Container([
    dbc.Row([
        html.Div('My First App with Data, Graph, and Controls', className="text-primary text-center fs-3")
    ]),

    dbc.Row([
            dcc.DatePickerRange(
                            id='date1',
                            min_date_allowed=date(2024, 2, 1),
                            max_date_allowed=date(2024, 2, 27),
                            initial_visible_month=date(2024, 2, 15),
                            end_date=date(2024, 2, 16),
                            #with_portal=True
                        ),
    ]),

    dbc.Row([

    html.Div([

        html.Div([
            dcc.Dropdown(
                cols,
                'herrenhausen_Temperatur',
                id='yaxis_col1'
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                cols,
                'herrenhausen_Feuchte',
                id='yaxis_col2'
            ),
   
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),
        
      dcc.Graph(figure={}, id='double_ax_graph'),

    ]),

    dbc.Row([

        dbc.Col([
            dbc.Row([
            
            dcc.Dropdown(cols, 'herrenhausen_Temperatur', multi=True,
                         id='radio-buttons-final'),
                         
                         


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
    Output(component_id='double_ax_graph', component_property='figure'),




    Input(component_id='radio-buttons-final', component_property='value'),

    Input('date1', 'start_date'),
    Input('date1', 'end_date'),

    Input('yaxis_col1', 'value'),
    Input('yaxis_col2', 'value'),

)
def update_graph(col_chosen_1,date1_start,date2_end,yaxis_col1,yaxis_col2):



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

    fig_dual_ax =  make_subplots(specs=[[{"secondary_y": True}]])



    dual_ax_filterd_df =  filtered_df[[yaxis_col1,yaxis_col2]]
        # Add traces
    fig_dual_ax.add_trace(
        go.Scatter(x=dual_ax_filterd_df.index, y= dual_ax_filterd_df[yaxis_col1], name="yaxis data"),
        secondary_y=False,
    )

    fig_dual_ax.add_trace(
        go.Scatter(x=dual_ax_filterd_df.index, y=dual_ax_filterd_df[yaxis_col2], name="yaxis2 data"),
        secondary_y=True,
    )
    # Add figure title
    fig_dual_ax.update_layout(
        title_text="Double Y Axis Example"
    )

    # Set x-axis title
    fig_dual_ax.update_xaxes(title_text="xaxis title")

    # Set y-axes titles
    fig_dual_ax.update_yaxes(title_text= dual_ax_filterd_df.columns[0], secondary_y=False)
    fig_dual_ax.update_yaxes(title_text=dual_ax_filterd_df.columns[1], secondary_y=True)


    return fig1,table1_columns, table1_data,fig_dual_ax

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
