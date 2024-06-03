from dash import html
from dash.dependencies import Input, Output
from dash import dcc, html
import dash_bootstrap_components as dbc
import xarray as xr
import pandas as pd
import datetime
from windrose import prepare_data_for_windrose
from dashboard_tools import calculate_trend
import plotly.express as px

#layout = #html.Div([
    #html.H3('Inhalt von Tab 1'),
    #dcc.Input(id='input-tab-1', value='Initial Value', type='text'),
    #html.Div(id='output-tab-1')
#])

date1 = datetime.datetime(2024, 2, 15, 0)
date2 = datetime.datetime(2024, 2, 15,23)

df = xr.open_dataset("dashboard/test_daten/2024-02_herrenhausen.nc").to_dataframe().loc[date1:date2]
df = df.assign(Date=df.index)

print(df["herrenhausen_Temperatur"])
layout = dbc.Container([
    dbc.Row([ #Erste Reihe

      
        dbc.Col([ # Erste Reihe, Erster Spot, Temp, Tmax, Tmin,
        html.Div(max(df["herrenhausen_Temperatur"])),
        html.Div(df["herrenhausen_Temperatur"].iloc[-1]),

        html.Div(min(df["herrenhausen_Temperatur"])),
        ]),
        dbc.Col([

        html.Div(str(df["herrenhausen_Druck"].iloc[-1])+ "h Pa"),
        html.Div(calculate_trend(df=df)),
            html.Div('Inhalt von 1 2')
        ])
        ,
        dbc.Col([
            html.Div('Inhalt von 1 3')
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.Div('Inhalt von 2 1')
        ])
        ,
        dbc.Col([
            html.Div('Inhalt von 2 2')
        ])
        ,
        dbc.Col([
            html.Div('Inhalt von 2 3')
        ])


    ]),

    dbc.Row([
        
        dbc.Col([
        html.Div('Inhalt von 3 1'),
        dcc.Input(id='input-tab-1', value='Initial Value', type='text'),

        ])
        ,
        dbc.Col([
        html.Div('Inhalt von 3 2'),
            dcc.Graph(figure={}, id='Windrose'),
           #html.Div( px.bar_polar(prepare_data_for_windrose(df), r='frequency', theta='wind_dir_bin', color='wind_speed_bin',
            #       color_discrete_sequence=px.colors.sequential.Plasma_r,
             #      title='Windrose Diagramm')),
        ])
        ,
        dbc.Col([
        html.Div('Inhalt von 3 3')
        ])


    ]),

    dbc.Row([
            dcc.Graph(figure={}, id='large Graphic'),


    ]),

])

def register_callbacks(app):
    @app.callback(
        [
        #Output('Temperatur','children'),
        Output('Windrose','figure'),
        Output("large Graphic", "figure")
        ],
        [Input('input-tab-1', 'value')]
    )
    def update_output(value):

        temperatur = df["herrenhausen_Temperatur"][:-1]
        print(temperatur)
        windrose_plot = px.bar_polar(prepare_data_for_windrose(df), r='frequency', theta='wind_dir_bin', color='wind_speed_bin',
                   color_discrete_sequence=px.colors.sequential.Plasma_r,
                   title='Windrose')

# Layout anpassen
        windrose_plot.update_layout(
        polar=dict(
            radialaxis=dict(ticks='', showticklabels=False),
            angularaxis=dict(direction='clockwise', showline=False)
        )
        )

        large_graphic = px.line(df["herrenhausen_Druck"])

        return windrose_plot,large_graphic
