import plotly.graph_objs as go
import plotly.io as pio
import json

# Daten für die Grafik
x = [1, 2, 3, 4, 5]
y = [10, 11, 12, 13, 14]

# Erstellen des Plotly-Traces
trace = go.Scatter(x=x, y=y)

# Erstellen des Layouts
layout = go.Layout(title='Meine Plotly Grafik')

# Erstellen des Figure-Objekts
fig = go.Figure(data=[trace], layout=layout)

# Konvertiere die Plotly-Grafik in JSON
fig_json = pio.to_json(fig)

# Speichere das JSON-Objekt in einer Datei
with open('meine_grafik.json', 'w') as json_file:
    json.dump(fig_json, json_file)


fig.show()
