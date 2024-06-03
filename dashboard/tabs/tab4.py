from dash import html
from dash.dependencies import Input, Output
from dash import dcc, html

layout = html.Div([
    html.H3('Inhalt von Tab 4'),
    dcc.Input(id='input-tab-4', value='Initial Value', type='text'),
    html.Div(id='output-tab-4')
])

def register_callbacks(app):
    @app.callback(
        Output('output-tab-4', 'children'),
        [Input('input-tab-4', 'value')]
    )
    def update_output(value):
        return f'Du hast eingegeben: {value}'
