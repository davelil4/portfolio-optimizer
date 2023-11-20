from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import data_grab as log

# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')


app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1(children='Minimum Variance Portfolio Optimizer', style={'textAlign':'center'}))),
        # dcc.Dropdown(df.country.unique(), 'Canada', id='dropdown-selection'),
        dbc.Row(
            dbc.Col(dbc.Input(
                id="ticker-asset",
                type="text",
                value='MSFT'
            ))),
        dbc.Row(
            dbc.Col(
                dcc.Graph(id='graph-content')
            )),
    ])

@callback(
    Output('graph-content', 'figure'),
    Input('ticker-asset', 'value')
)
def update_graph(value):
    dff = log.getHistory(value)
    return px.line(dff, y='Close', title=value + " Close Stock Price")

if __name__ == '__main__':
    app.run(debug=True)
