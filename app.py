from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
import data_grab as dg
import layout as lay
import linprog as lp

# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')


app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

test_syms = ["AAPL", "AEP", "BAX", "ED", "F", "GE", "GOOG", "MCD", "MSFT"]
port = lp.getEffPort(test_syms)

app.layout = dbc.Container(
    [
        dcc.Store(id='local', storage_type='local'),
        dbc.Row(
            dbc.Col(
                html.H1(children='Portfolio Optimizer', style={'textAlign':'center'}))),
        # dcc.Dropdown(df.country.unique(), 'Canada', id='dropdown-selection'),
        dbc.Row(
            dbc.Col(dcc.Dropdown(
                id="ticker-asset",
                options=dg.stock_symbols,
                placeholder="Select a Stock",
                style={"color": "black"}
            ), width=6)),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id='graph-content', figure=px.line(dg.getHistory('MSFT'), y='Close', title="MSFT Close Stock Price"))
                ),
                dbc.Col(
                    dcc.Graph(id='port-graph', figure=lay.build_port_figure(test_syms, port))
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Stack([
                        dbc.Label("Budget:"),
                        dbc.Input(id='budget', type="number", persistence=True, persistence_type='local')
                    ], direction="horizontal", gap=3),
                    width=2
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    id='port-table',
                )
            ]
        )
    ])

@callback(
    Output('graph-content', 'figure'),
    Input('ticker-asset', 'value')
)
def update_close_ticker(value):
    if value is None:
        raise PreventUpdate
    dff = dg.getHistory(value)
    return px.line(dff, y='Close', title=value + " Close Stock Price")

@callback(
    Output('local', 'data'),
    Input('budget', 'value')
)
def update_budget(value):
    if value is None:
        raise PreventUpdate
    
    return {'budget': value}

@callback(
    Output('port-table', 'children'),
    Input('local', 'data'),
)
def update_port_table(data):
    if data is None:
        raise PreventUpdate
    
    budget = data['budget']
    return lay.build_port_table(test_syms, port, budget)
    

if __name__ == '__main__':
    app.run(debug=True)
