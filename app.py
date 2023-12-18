from dash import Dash, html, dcc, callback, Output, Input, State, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
import data_grab as dg
import layout as lay
import linprog as lp
import modeling as ml

# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')


app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

test_syms = ["AAPL", "AEP", "BAX", "ED", "F", "GE", "GOOG", "MCD", "MSFT"]
port = lp.getEffPort(test_syms)

app.layout = dbc.Container(
    [
        dcc.Store(id='local', storage_type='local'),
        dbc.Row(
            dbc.Col(
                html.H1(children='Portfolio Optimizer', style={'textAlign':'center'})
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Stack([
                            dcc.Dropdown(id="ticker-asset",options=list(dg.stock_symbols),placeholder="Select a Stock", style={"color": "black"}, value="MSFT"),
                            dcc.Dropdown(id="ticker-chooser", options=['Close', 'LogReturns', 'Open'], persistence=True, persistence_type='session', style={"color": "black"}, value='Close'),
                            dcc.Dropdown(id="ticker-plot-chooser", options=['Line', 'Histogram'], persistence=True, persistence_type='session', style={"color": "black"}, value='Line'),
                            dcc.Dropdown(id="ticker-time-chooser", options=['1d', '5d', '1mo', '3mo', '6mo', '1y', 'ytd', 'max'], persistence=True, persistence_type='session', style={"color": "black"}, value='1mo')
                            ], direction="horizontal", gap=3),
                        
                        dcc.Graph(id='graph-content')
                    ]
                ),
                dbc.Col(
                    [
                        dcc.Dropdown(),
                        dcc.Graph(id='port-graph', figure=lay.build_port_figure(test_syms, port))
                    ]
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
                ),
                # dbc.Col(
                #     dcc.Graph(figure=ml.predictStock("MSFT"))
                # )
            ]
        )
    ])

@callback(
    Output('graph-content', 'figure'),
    Input('ticker-asset', 'value'),
    Input('ticker-chooser', 'value'),
    Input('ticker-plot-chooser', 'value'),
    Input('ticker-time-chooser', 'value'),
    
)
def update_close_ticker(symbol, choice, plot_choice, time_span):
    if symbol is None:
        raise PreventUpdate
    dff = dg.getHistory(symbol, time_span)
    
    if plot_choice == 'Histogram':
        return px.histogram(dff, x=choice, title=symbol + " " + choice)
    
    return px.line(dff, y=choice, title=symbol + " " + choice)

@callback(
    Output('port-table', 'children'),
    Output('port-graph', 'figure'),
    Input('budget', 'value'),
    Input('lower-bound', 'value'),
    Input('upper-bound', 'value'),
    Input('local', 'data'),
)
def update_from_bounds(budg, lb, ub, data):

    if budg is None:
        raise PreventUpdate
    
    port = None
    
    syms = test_syms
    
    if "symbols" in data: syms = data["symbols"]
    
    if lb is not None and ub is not None:
        port = lp.getEffPort(syms, lb, ub)
    elif lb is not None:
        port = lp.getEffPort(syms, lb)
    elif ub is not None: 
        port = lp.getEffPort(syms, ub)
    else: 
        port = lp.getEffPort(syms, budg)
    
    
    return lay.build_port_table(syms, port, budg), lay.build_port_figure(syms, port)

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
    if ctx.triggered[0]['prop_id'] == 'add-button.n_clicks' and symbol in dg.stock_symbols:
        if "symbols" not in data:
            data["symbols"] = []
        data["symbols"].append(symbol)
    else:
        if symbol in data["symbols"]:
            data["symbols"].remove(symbol)
            if len(data["symbols"]) == 0:
                del data["symbols"]
    
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
