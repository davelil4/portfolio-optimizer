from dash import callback, Output, Input, State, html, dcc
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import data_grab as dg
import ml_modeling.modeling as ml
from ml_modeling.layout import *
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# from icecream import ic
from ml_modeling.helpers import *

def make_layout():
    return html.Div([html.H2('Simulations'),
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
                            dcc.Dropdown(options=dg.stock_symbols, id='sim_stock'),
                            dbc.Button('Run Simulations', 'b_sims', color='primary'),
                        ], direction='horizontal', gap=3)
                    ], width=6),
                    dbc.Col([
                        dbc.Stack([
                            dbc.Label(id='sim_prec'),
                            dbc.Label(id='sim_grets')
                        ], direction='horizontal', gap=3),
                    ], width=6)
                ])
            ])

def make_callbacks():
    @callback(
        [
            Output('MCSims', 'figure'),
            Output('strat', 'figure'),
            Output('ticker_data', 'data', allow_duplicate=True),
            Output('sim_data', 'data')
        ],
        [
            Input('b_sims', 'n_clicks'),
            State('nsims', 'value'),
            State('ndays', 'value'),
            State('sim_stock', 'value'),
            State('ticker_data', 'data')
        ],
        background=True,
        running=[
            (Output('b_sims', 'disabled'), True, False),
        ],
        prevent_initial_call=True
    )
    def simulations(b_sims, nsims, ndays, symbol, data):
        if not b_sims or not nsims or not ndays or not symbol:
            raise PreventUpdate
        
        hist, data = get_hist(data, symbol)
        
        res = ml.runSims(
            hist, 
            RandomForestClassifier(random_state=1),
            ['Open', 'Close', 'High', 'Low'],
            ndays,
            nsims
        )
        
        imp = {
            'precision': res['precision'], 
            'grets_avg': res['grets_avg']
        }
        
        return ml.simFigure(hist, res['data']), ml.stratFigure(hist, res, nsims, ndays), dict(data), imp
        
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
        
        score = data['precision']
        rets = data['grets_avg']
        
        return f'Precision: {score:.4f}', f'Gross Returns: {rets:.4f}'