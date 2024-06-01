from dash import html, dcc, callback, Output, Input, State, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import data_grab as dg
import ml_modeling.modeling as ml
from ml_modeling.layout import *
from datetime import datetime
import ml_modeling.model_finding as mf
# from icecream import ic
import joblib as jb
from sklearn.metrics import precision_score
import ml_modeling.indicator_testing as it
import ml_modeling.indicator_graphing as ig
import ml_modeling.simulations as sims
from ml_modeling.helpers import *
from ml_modeling.model_finding import models
import ml_modeling.model_selection as ms

og_preds = ['Open', 'Close', 'High', 'Low']

ml_tab = html.Div(
    [
        dcc.Store(id='sim_data', storage_type='memory'),
        dcc.Store(id='models', storage_type='local'),
        dcc.Store(id='indicator-df', storage_type='memory'),
        dbc.Card(
            dbc.CardBody([
                html.H2('LOREM IPSUM'),
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(['Model Selection', 'Backtesting', 'Indicators'], 'Model Selection', id='dd_ms_graph', persistence=True),
                        html.Br(),
                        html.Div([
                            dcc.Graph('ms_graph'),
                            html.Br(),
                            dbc.Button('Run Model Selection', id='b_ms'),
                        ], id='ms_comps'),
                        html.Div([
                            dcc.Graph('bt_graph'),
                            html.Br(),
                            dbc.Button('Run Backtest', id='b_bt'),
                            dbc.Label(id='bt_prec')
                        ], id='bt_comps'),
                        ig.make_layout(),
                        html.Br(),
                        dbc.Stack([dbc.Label('Stock Data'), dcc.Dropdown(dg.stock_symbols, id='dd_ms', style={'width': '40%'}),], direction='horizontal', gap=3)
                    ]),
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        ms.make_layout() 
                    ], width=6),
                    dbc.Col([
                        it.make_layout()
                    ], width=6)
                ])
            ])
        ),
        html.Br(),
        dbc.Card(
            dbc.CardBody([
                sims.make_layout()
            ])
        ),
    ],
    id='ml_tab'
)

def create_date(string):
    return datetime.strptime(string, '%Y-%M-%D').date()

def model_from_inputs(model_name, input_dicts, values):
    params = list(map(lambda x: x['index'], input_dicts))
    param_map = {params[i]: values[i] for i in range(len(values))}
    return models[model_name](**param_map)

# def create_training(hist, cl_preds, shift):
#     cl_preds = [] if not cl_preds else cl_preds
#     train = ml.create_shifted_data(hist, shift)
#     train = ml.create_new_predictors(train, cl_preds)
    
#     return train

def create_training(hist, indicators, shift):
    train = ml.create_shifted_data(hist, shift)
    return it.gen_indicators(train, indicators)

@callback(
    
    [
        Output('ms_graph', 'figure'),
        Output('ticker_data', 'data', allow_duplicate=True)
    ],
    [
        Input('b_ms','n_clicks'),
        State('ticker_data', 'data'),
        State('dd_ms', 'value'),
        State('ind-store', 'data')
    ],
    background=True,
    running=[
        (Output('b_ms', 'disabled'), True, False),
    ],
    prevent_initial_call=True
)
def run_model_selection(b_ms, data, symbol, inds):
    if not b_ms or not symbol:
        raise PreventUpdate

    hist, data = get_hist(data, symbol)
    
    train = create_training(hist, inds, 1).dropna()
    
    return mf.drawMSFigure(*mf.model_selection(train, it.get_indicators(inds) + og_preds)), dict(data)


@callback(
    [
        Output('ms_comps', 'hidden'),
        Output('bt_comps', 'hidden'),
        Output('ind_comps', 'hidden')
    ],
    Input('dd_ms_graph', 'value'),
)
def model_graph(dd):
    if dd == 'Model Selection':
        return False, True, True
    elif dd == 'Backtesting':
        return True, False, True
    return True, True, False



@callback(
    [
        Output('bt_graph', 'figure'),
        Output('bt_prec', 'children'),
        Output('ticker_data', 'data', allow_duplicate=True)
    ],
    [
        Input('b_bt', 'n_clicks'),
        State('ticker_data', 'data'),
        State('dd_ms', 'value'),
        State('ind-store', 'data'),
    ],
    background=True,
    running=[
        (Output('b_bt', 'disabled'), True, False),
    ],
    prevent_initial_call=True
)
def backtest_model(b_bt, ticker_data, symbol, inds):
    if b_bt is None: raise PreventUpdate
    
    hist, data = get_hist(ticker_data, symbol)
    
    b_data = create_training(hist, inds, 1).dropna()
    
    res = ml.backtest(
        b_data,
        jb.load('model.joblib'), 
        it.get_indicators(inds) + og_preds
    )
    
    prec = precision_score(res['Target'], res['Predictions'])
    
    return res.plot(backend='plotly'), f'Precision: {prec:.3f}', dict(data)


@callback(
    Output('indicator-df', 'data'),
    Input('ticker_data', 'data'),
    Input('ind-store', 'data'),
    State('dd_ms', 'value')
)
def update_data(hist_data, inds, symbol):
    if not hist_data or not symbol:
        raise PreventUpdate
    hist, _ = get_hist(hist_data, symbol)
    inds = it.gen_indicators(hist, inds)
    return create_data_from_df(inds, symbol)

it.make_callbacks()

ig.make_callbacks()

sims.make_callbacks()

ms.make_callbacks()