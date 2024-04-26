from dash import html, dcc, callback, Output, Input, State, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
# import plotly.express as px
# import pandas as pd
import data_grab as dg
import ml_modeling.modeling as ml
from ml_modeling.layout import *
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import ml_modeling.model_selection as ms
import inspect
# from icecream import ic
import joblib as jb
from sklearn.metrics import precision_score
import ml_modeling.indicator_testing as it
import ml_modeling.indicator_graphing as ig
from ml_modeling.helpers import *


models = {
    'RandomForestClassifier': RandomForestClassifier
}

og_preds = ["Open", "Close", "High", "Low"]

ml_tab = html.Div(
    [
        dcc.Store(id='sim_data', storage_type='memory'),
        dcc.Store(id='model', storage_type='memory'),
        dcc.Store(id='indicator-df', storage_type='memory'),
        dbc.Card(
            dbc.CardBody([
                html.H2("Model Selection"),
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(['Model Selection', 'Backtesting', 'Indicators'], 'Model Selection', id='dd_ms_graph', persistence=True),
                        html.Br(),
                        html.Div([
                            dcc.Graph('ms_graph'),
                            html.Br(),
                            dbc.Button('Run Model Selection', id='b_ms'),
                        ], "ms_comps"),
                        html.Div([
                            dcc.Graph('bt_graph'),
                            html.Br(),
                            dbc.Button('Run Backtest', id='b_bt'),
                            dbc.Label(id='bt_prec')
                        ], id="bt_comps"),
                        ig.make_layout(),
                        html.Br(),
                        dcc.Dropdown(dg.stock_symbols, id='dd_ms', style={'width': '40%'}),
                    ]),
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col([
                       html.Div([
                            html.H5("Model"),
                            dcc.Dropdown(list(models.keys()),id='model_select'),
                            html.Div(id='model_params')
                            ]),
                            html.Br(),
                            dbc.Stack([
                                dbc.Button("Save Model", "b_save"),
                                dbc.Label(id='l_succ')
                        ], direction='horizontal', gap=3) 
                    ], width=6),
                    dbc.Col([
                    it.make_layout(),
                    ], width=6)
                ])
            ])
        ),
        html.Br(),
        dbc.Card(
            dbc.CardBody([
                html.H2("Simulations"),
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
            ])
        ),
    ],
    id='ml_tab'
)
    

def get_hist(data, symbol):
    if data is None or (pd.to_datetime(data['last_date']).date() < pd.Timestamp.today("America/New_York").date()) or symbol not in data or 'last_date' not in data:
        hist = dg.getHistory(symbol, 'max')
        data = create_data_from_df(hist, symbol)
    else:
        hist = create_df_from_data(data, symbol)
    
    return hist, data

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
    prevent_initial_call=True
)
def simulations(b_sims, nsims, ndays, symbol, data):
    if not b_sims or not nsims or not ndays or not symbol:
        raise PreventUpdate
    
    hist, data = get_hist(data, symbol)
    
    res = ml.runSims(
        hist, 
        RandomForestClassifier(random_state=1),
        ["Open", "Close", "High", "Low"],
        ndays,
        nsims
    )
    
    imp = {
        "precision": res["precision"], 
        "grets_avg": res["grets_avg"]
    }
    
    return ml.simFigure(hist, res["data"]), ml.stratFigure(hist, res, nsims, ndays), dict(data), imp
    
    
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
    
    return f"Precision: {score:.4f}", f"Gross Returns: {rets:.4f}"

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
        (Output("b_ms", "disabled"), True, False),
    ],
    prevent_initial_call=True
)
def run_model_selection(b_ms, data, symbol, inds):
    if not b_ms or not symbol:
        raise PreventUpdate

    hist, data = get_hist(data, symbol)
    
    train = create_training(hist, inds, 1).dropna()
    
    return ms.drawMSFigure(*ms.model_selection(train, it.get_indicators(inds) + og_preds)), dict(data)

@callback(
    Output('model_params', 'children'),
    Input('model_select', 'value')
)
def create_model_params(model_name):
    
    def create_param(arg, default):
        return dbc.Stack(
                [
                    dbc.Label(f"{arg}: "),
                    dbc.Input({"type": "model_param", "index": arg}, value=default)
                ], direction='horizontal', gap=3
            )
    
    if model_name is None:
        raise PreventUpdate
    
    
    model = models[model_name]
    
    args1, _, _, arg_def, _, kwargs, *_ = inspect.getfullargspec(model.__init__)
    
    params = []
    
    for i, arg in enumerate(args1[1:]):
        
        params.append(
            create_param(arg, arg_def[i])
        )
    
    for arg in kwargs:
        params.append(
            create_param(arg, kwargs[arg])
        )
    
    return params

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
        True, False, True
    else:
        return True, True, False


@callback(
    [
        Output('model', 'data'),
        Output('l_succ', 'children')
    ],
    [
        Input('b_save', 'n_clicks'),
        State('model_select', 'value'),
        State({'type': 'model_param', 'index': ALL}, 'value'),
        State({'type': 'model_param', 'index': ALL}, 'id')
    ]
)
def save_model(b_save, model_name, vals, ids):
    if b_save is None:
        raise PreventUpdate
    
    jb.dump(model_from_inputs(model_name, ids, vals), 'model.joblib')
    
    return [model_name, ids, vals], "Successfully saved model."


@callback(
    [
        Output('bt_graph', 'figure'),
        Output('bt_prec', 'children'),
        Output('ticker_data', 'data', allow_duplicate=True)
    ],
    [
        Input('b_bt', 'n_clicks'),
        # State('model', 'data'),
        State('ticker_data', 'data'),
        State('dd_ms', 'value'),
        State('ind-store', 'data'),
    ],
    background=True,
    running=[
        (Output("b_bt", "disabled"), True, False),
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
    
    prec = precision_score(res["Target"], res["Predictions"])
    
    return res.plot(backend='plotly'), f"Precision: {prec:.3f}", dict(data)


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