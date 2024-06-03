import dash
from dash import Dash, html, dash_table, dcc, callback, Output, Input
from dash.dependencies import Input, Output
from tabs import tab1, tab2, tab3, tab4
import dash_bootstrap_components as dbc
import plotly.express as px

external_stylesheets = [dbc.themes.CERULEAN]
app = dash.Dash(__name__, suppress_callback_exceptions=True,external_stylesheets=external_stylesheets)
app.title = "Mehrere Tabs Beispiel"

# Register the layouts of each tab
tab1.register_callbacks(app)
tab2.register_callbacks(app)
tab3.register_callbacks(app)
tab4.register_callbacks(app)

app.layout = html.Div([
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Heute', value='tab-1'),
        dcc.Tab(label='Temperatur und Feuchte', value='tab-2'),
        dcc.Tab(label='Niederschlag und Wolken', value='tab-3'),
        dcc.Tab(label='Baukasten', value='tab-4'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return tab1.layout
    elif tab == 'tab-2':
        return tab2.layout
    elif tab == 'tab-3':
        return tab3.layout
    elif tab == 'tab-4':
        return tab4.layout

if __name__ == '__main__':
    app.run_server(debug=True)
