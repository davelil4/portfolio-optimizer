from dash import Dash, html, dcc, callback, Output, Input, State, callback_context
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
                        dbc.Input(id='budget', placeholder="Budget", type="number", persistence=True, persistence_type='local', debounce=True)
                    ], direction="horizontal", gap=3),
                    width=2
                ),
                dbc.Col(
                    dbc.Stack([
                        dbc.Label("Bounds:"),
                        dbc.Input(id='lower-bound', placeholder="Lower Bound", type="number", persistence=True, persistence_type='local', debounce=True, value=0),
                        dbc.Input(id='upper-bound', placeholder="Upper Bound", type="number", persistence=True, persistence_type='local', debounce=True),
                    ], direction="horizontal", gap=3),
                    width=3
                ),
                dbc.Col(
                    dbc.Stack([
                        dbc.Label("Edit Portfolio:"),
                        dbc.Input(id="port-edit", placeholder="Stock symbol to add/remove...", type="text", style={"width": "20%"}),
                        dbc.Button("Add", id="add-button", color="primary"),
                        dbc.Button("Remove", id="remove-button", color="primary"),
                        dbc.Label(id="symbols")
                    ], direction="horizontal", gap=3),
                    width=6
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
    Output('port-table', 'children'),
    Output('port-graph', 'figure'),
    Input('budget', 'value'),
    Input('lower-bound', 'value'),
    Input('upper-bound', 'value'),
)
def update_from_bounds(budg, lb, ub):

    if budg is None:
        raise PreventUpdate
    
    port = None
    
    if lb is not None and ub is not None:
        port = lp.getEffPort(test_syms, lb, ub)
    elif lb is not None:
        port = lp.getEffPort(test_syms, lb)
    elif ub is not None: 
        port = lp.getEffPort(test_syms, ub)
    else: 
        port = lp.getEffPort(test_syms, budg)
    
    return lay.build_port_table(test_syms, port, budg), lay.build_port_figure(test_syms, port)

@callback(
    Output('lower-bound', 'invalid'),
    Input('lower-bound', 'value'),
)
def lower_bound_invalidation(val):
    if val is None:
        raise PreventUpdate

    if val > 0:
        return True
    return False

@callback(
    Output('local', 'data'),
    Input('add-button', 'n_clicks'),
    Input('remove-button', 'n_clicks'),
    State('local', 'data'),
    State('port-edit', 'value')
)
def update_symbols(add, remove, data, symbol):
    if add is None and remove is None:
        raise PreventUpdate

    data = data if data is not None else {}
    
    ctx = callback_context
    if ctx.triggered[0]['prop_id'] == 'add-button.n_clicks':
        if "symbols" not in data:
            data["symbols"] = []
        data["symbols"].append(symbol)
    else:
        if symbol in data["symbols"]:
            data["symbols"].remove(symbol)
    
    return data

@callback(
    Output('symbols', 'children'),
    Input('local', 'data')
)
def update_symbols_label(data):
    if data is None or "symbols" not in data:
        raise PreventUpdate
    
    return ", ".join(data["symbols"])

if __name__ == '__main__':
    app.run(debug=True)
