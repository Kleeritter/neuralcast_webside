from dash import html
from dash.dependencies import Input, Output,State
from dash import dcc, html
import dash_bootstrap_components as dbc
import xarray as xr
import pandas as pd
import datetime
from windrose import prepare_data_for_windrose
from dashboard_tools import calculate_trend
import plotly.express as px
from dash_breakpoints import WindowBreakpoints
import dash_daq as daq

#layout = #html.Div([
    #html.H3('Inhalt von Tab 1'),
    #dcc.Input(id='input-tab-1', value='Initial Value', type='text'),
    #html.Div(id='output-tab-1')
#])
drop_shadow_style = {
    "boxShadow": "0 4px 8px 0 rgba(0, 0, 0, 0.2)",  # Drop-Shadow hinzufügen
    "transition": "0.3s"
}
date1 = datetime.datetime(2024, 2, 15, 0)
date2 = datetime.datetime(2024, 2, 15,23)

df = xr.open_dataset("dashboard/test_daten/2024-02_herrenhausen.nc").to_dataframe().loc[date1:date2]
df = df.assign(Date=df.index)

#print(df["herrenhausen_Temperatur"])
layout = dbc.Container([

    dcc.Interval(
    id="load_interval", 
    n_intervals=0, 
    max_intervals=0, #<-- only run once
    interval=1
),

    dbc.Row([ #Erste Reihe

      
        dbc.Col([ # Erste Reihe, Erster Spot, Temp, Tmax, Tmin,
        dbc.Card(
        html.Div([
        html.Div(max(df["herrenhausen_Temperatur"])),
        html.H2(df["herrenhausen_Temperatur"].iloc[-1]),
            
        html.Div(min(df["herrenhausen_Temperatur"])),
        ]),
            className="mb-3",

        
        style={**drop_shadow_style,"height":"20vh",'textAlign': 'center'} )
        ],md=4),
        dbc.Col([
            
            dbc.Card(
        html.Div([
        html.Div(  
        html.H2(str(df["herrenhausen_Druck"].iloc[-1])+ " h Pa"), style={  'position': 'absolute',
  'top': '0',
  'left': '0',
  'z-index': '10',
  'textAlign': 'center'}),
        html.Div(
         dcc.Graph(figure={}, id='overpres',style={"height": "30vh"},config={'staticPlot': True})
         ,style={'position': 'relative'}
         
         ),
        #html.Div(calculate_trend(df=df)),
            ]),
                        className="mb-3",
            style={**drop_shadow_style,"height":"20vh"} ),
        ],md=4)
        ,
        dbc.Col([
            dbc.Card(
            html.Div(html.Img(src="assets/partly-cloudy-day.svg")),
                        className="mb-3",
             style={**drop_shadow_style,"height":"12vh"} ),
        ],md=4)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card(
            html.Div(   html.H2(str(df["herrenhausen_Regen"].iloc[-1])+ "mm"),style={'textAlign': 'center'} ),
            
                        className="mb-3",
             style={**drop_shadow_style,"height":"12vh"} )
        ],md=4)
        ,
        dbc.Col([
             dbc.Card(
            html.Div(   html.H2(str(df["herrenhausen_Feuchte"].iloc[-1])+ "%"),style={'textAlign': 'center'} ),
            
                        className="mb-3",
             style={**drop_shadow_style,"height":"12vh"} )
        ],md=4)
        ,
        dbc.Col([
            dbc.Card(
            html.Div(   html.H2(str(df["dach_Global_CMP-11"].iloc[-1])+ "W/m^2"),style={'textAlign': 'center'} ),
            
                        className="mb-3",
             style={**drop_shadow_style,"height":"12vh"} )
            
        ],md=4)


    ]),

    dbc.Row([
        

        dbc.Col([
            dbc.Card(
            dcc.Graph(figure={}, id='Windrose',style={"height": "30vh"}),
           #html.Div( px.bar_polar(prepare_data_for_windrose(df), r='frequency', theta='wind_dir_bin', color='wind_speed_bin',
            #       color_discrete_sequence=px.colors.sequential.Plasma_r,
             #      title='Windrose Diagramm')),
              style={"height":"31vh"} )
        ]
        ,md=8)
        ,
        dbc.Col([
            dbc.Card(
         [
         html.Div(html.Img(src="assets/place.png"))],
        style={"height":"100%"} 
            )
        ],md=4)


    ]),

    dbc.Row([
        #dbc.Card(
            dcc.Graph(figure={}, id='large Graphic'),
        #)

    ]),

] ,fluid=True)





def register_callbacks(app):
    @app.callback(

        [
        #Output('Temperatur','children'),
        Output('Windrose','figure'),
        Output("large Graphic", "figure"),
        Output("overpres", "figure")
        ],
             [       Input(component_id="load_interval", component_property="n_intervals"),
],   # State("breakpoints", "width"),

    )
    def update_output(width):

        temperatur = df["herrenhausen_Temperatur"][:-1]
        #print(temperatur)
        show_legend = True
        if  width < 768:  # Schwellenwert für kleine Bildschirme (768px)
            show_legend = False

        windrose_plot = px.bar_polar(prepare_data_for_windrose(df), r='frequency', theta='wind_dir_bin', color='wind_speed_bin',
                   color_discrete_sequence=px.colors.sequential.Plasma_r,
                   )

# Layout anpassen
        windrose_plot.update_layout(
        showlegend=show_legend,  # Legende basierend auf Bildschirmgröße anzeigen oder ausblenden
        polar=dict(
            radialaxis=dict(ticks='', showticklabels=False),
            angularaxis=dict(direction='clockwise', showline=False)
        ),
        legend=dict(
    orientation="h",)
        )

        large_graphic = px.line(df["herrenhausen_Druck"])
        over_pres = px.line(df["herrenhausen_Druck"][-180:])
        over_pres.update_layout(showlegend=False)
        over_pres.update_xaxes(showticklabels=False,showgrid=False,visible=False) # Hide x axis ticks 
        over_pres.update_yaxes(showticklabels=False,showgrid=False,visible=False) # Hide y axis ticks
        #over_pres.update_config({'staticPlot': True})


        return windrose_plot,large_graphic,over_pres
