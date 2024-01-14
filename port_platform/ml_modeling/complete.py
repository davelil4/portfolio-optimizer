from dash import html, dcc, callback, Output, Input, State, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.express as px
# import pandas as pd
import data_grab as dg
import ml_modeling.modeling as ml
from ml_modeling.layout import *
from datetime import date
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from icecream import ic

ml_tab = html.Div(
    [
        dcc.Store(id='sim_data', storage_type='session'),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id='MCSims'), width='6'
            ),
            dbc.Col(
                dcc.Graph(id='strat'), width='6'
            )
        ]),
        html.Br(),
        dbc.Row([
            dbc.Col([
                dbc.Stack([
                    drawSimInputs(),
                    dcc.Dropdown(options=dg.stock_symbols, id="sim_stock"),
                    dbc.Button("Run Simulations", "b_sims", color="primary"),
                ], direction='horizontal', gap=3)
            ], width=6),
            dbc.Col([
                dbc.Stack([
                    dbc.Label(id="sim_prec"),
                    dbc.Label(id="sim_grets")
                ], direction='horizontal', gap=3),
            ], width=6)
        ])
    ],
    id='ml_tab'
)

def createDate(string):
    return datetime.strptime(string, '%Y-%M-%D').date()

@callback(
    [
        Output('MCSims', 'figure'),
        Output('strat', 'figure'),
        Output('ticker_data', 'data'),
        # Output('sim_data', 'data')
    ],
    [
        Input('b_sims', 'n_clicks'),
        State('nsims', 'value'),
        State('ndays', 'value'),
        State('sim_stock', 'value'),
        State('ticker_data', 'data')
    ]
)
def simulations(b_sims, nsims, ndays, symbol, data):
    if not b_sims or not nsims or not ndays or not symbol:
        raise PreventUpdate
    
    hist = None
    if not data or (pd.to_datetime(data['last_date']).date() < pd.Timestamp.today("America/New_York").date()):
        data = {}
        hist = dg.getHistory(symbol, 'max')
        hist['Date'] = hist.index
        data[symbol] = hist.to_dict('records')
        del hist['Date']
        data['last_date'] = hist.iloc[[-1]].index[0]
    else:
        hist = pd.DataFrame.from_dict(data[symbol])
        hist.set_index('Date')
        del hist['Date']
    
    res = ml.runSims(
        hist, 
        RandomForestClassifier(random_state=1),
        ["Open", "Close", "High", "Low"],
        ndays,
        nsims
    )
    
    return ml.simFigure(hist, res["data"]), ml.stratFigure(hist, res, nsims, ndays), dict(data)
    
    
@callback(
    [
        Output('sim_prec', 'children'),
        Output('sim_grets', 'children')
    ],
    [
        Input('sim_data', 'data')
    ]
)
def simulated_precision(data):
    if data is None:
        raise PreventUpdate
    
    score = data["precision"]
    rets = data["grets_avg"]
    
    return f"Precision: {score}", f"Grets: {rets}"
    