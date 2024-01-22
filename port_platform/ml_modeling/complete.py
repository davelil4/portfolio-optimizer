from dash import html, dcc, callback, Output, Input, State, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.express as px
# import pandas as pd
import data_grab as dg
import ml_modeling.modeling as ml
from ml_modeling.layout import *
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import ml_modeling.model_selection as ms
import inspect
from icecream import ic
import joblib as jb
from sklearn.metrics import precision_score

models = {
    'RandomForestClassifier': RandomForestClassifier
}

ml_tab = html.Div(
    [
        dcc.Store(id='sim_data', storage_type='memory'),
        dcc.Store(id='model', storage_type='memory'),
        dbc.Card(
            dbc.CardBody([
                html.H2("Model Selection"),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H5("Indicators/Predictors"),
                            dcc.Checklist(
                                options=list(ml.pred_to_func.keys()),
                                id="cl_ex_preds"
                                # inline=True
                            ),
                            dcc.Checklist(
                                options=[
                                    "Open",
                                    "Close",
                                    "High",
                                    "Low"
                                ],
                                id="cl_og_preds"
                            ),
                        ]),
                        html.Br(),
                        html.Div([
                            html.H5("Model"),
                            dcc.Dropdown(list(models.keys()),id='model_select'),
                            html.Div(id='model_params')
                        ], style={'width': '50%'}),
                        html.Br(),
                        dbc.Stack([
                            dbc.Button("Save Model", "b_save"),
                            dbc.Label(id='l_succ')
                        ], direction='horizontal', gap=3)
                    ]),
                    dbc.Col([
                        dcc.Dropdown(['Model Selection', 'Backtesting'], 'Model Selection', id='dd_ms_graph'),
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
                        ], "bt_comps"),  
                        html.Br(),
                        dcc.Dropdown(dg.stock_symbols, id='dd_ms', style={'width': '30%'})
                    ])
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
        # html.Br(),
        # dbc.Card(
        #     dbc.CardBody([
        #         html.H2("Backtesting"),
        #         dbc.Row(
        #             dbc.Col(
        #                 dcc.Graph(id='predictions')
        #             )
        #         ),
        #         html.Br(),
        #         dbc.Row([
        #             dbc.Col(
        #                 dbc.Button("Run Backtesting", 'b_backtest')
        #             ),
        #             dbc.Col(
        #                 dbc.Label(id='back_prec')
        #             )
        #         ])
        #     ])
        # )
    ],
    id='ml_tab'
)

def get_hist(data, symbol):
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
    
    return hist

def createDate(string):
    return datetime.strptime(string, '%Y-%M-%D').date()

def model_from_inputs(model_name, input_dicts, values):
    params = list(map(lambda x: x['index'], input_dicts))
    param_map = {params[i]: values[i] for i in range(len(values))}
    return models[model_name](**param_map)

def createTraining(hist, cl_preds, shift):
    cl_preds = [] if not cl_preds else cl_preds
    train = ml.create_shifted_data(hist, shift)
    train = ml.create_new_predictors(train, cl_preds)
    
    return train

@callback(
    [
        Output('MCSims', 'figure'),
        Output('strat', 'figure'),
        Output('ticker_data', 'data'),
        Output('sim_data', 'data')
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
    
    hist = get_hist(data, symbol)
    
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
    
    Output('ms_graph', 'figure'),
    [
        Input('b_ms','n_clicks'),
        State('ticker_data', 'data'),
        State('dd_ms', 'value'),
        State('cl_ex_preds', 'value'),
        State('cl_og_preds', 'value')
    ],
    background=True,
    running=[
        (Output("b_ms", "disabled"), True, False),
    ],
)
def run_model_selection(b_ms, data, symbol, cl_preds, og_preds):
    if not b_ms or not data or not symbol:
        raise PreventUpdate

    hist = get_hist(data, symbol)
    
    cl_preds = [] if not cl_preds else cl_preds
    train = createTraining(hist, cl_preds, 1).dropna()
    
    
    # ic(train)
    
    return ms.drawMSFigure(*ms.model_selection(train, cl_preds + og_preds))


@callback(
    Output('model_params', 'children'),
    Input('model_select', 'value')
)
def create_model_params(model_name):
    
    def create_param(arg, default):
        return dbc.Stack(
                [
                    dbc.Label(f"{arg}: "),
                    dbc.Input({"type": "param", "index": arg}, value=default)
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
    ],
    Input('dd_ms_graph', 'value'),
)
def model_graph(dd):
    
    if dd == 'Model Selection':
        return False, True
    return True, False


@callback(
    [
        Output('model', 'data'),
        Output('l_succ', 'children')
    ],
    [
        Input('b_save', 'n_clicks'),
        State('model_select', 'value'),
        State({'type': 'param', 'index': ALL}, 'value'),
        State({'type': 'param', 'index': ALL}, 'id')
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
        Output('bt_prec', 'children')
    ],
    [
        Input('b_bt', 'n_clicks'),
        # State('model', 'data'),
        State('ticker_data', 'data'),
        State('dd_ms', 'value'),
        State('cl_ex_preds', 'value'),
        State('cl_og_preds', 'value')
    ],
    background=True,
    running=[
        (Output("b_bt", "disabled"), True, False),
    ],
)
def backtest_model(b_bt, ticker_data, symbol, cl_preds, og_preds):
    if b_bt is None: raise PreventUpdate
    
    hist = get_hist(ticker_data, symbol)
    cl_preds = [] if not cl_preds else cl_preds
    
    data = createTraining(hist, cl_preds, 1).dropna()
    
    res = ml.backtest(
        data,
        jb.load('model.joblib'), 
        cl_preds + og_preds
    )
    
    prec = precision_score(res["Target"], res["Predictions"])
    
    return res.plot(backend='plotly'), f"Precision: {prec:.3f}"

# @callback(
#     Output('b_backtest', 'style'),
#     [
#         State('b_backtest', 'style'),
#         Input('model_params', 'children'),
#         State({'type': 'param', 'index': ALL}, 'value'),
#         State({'type': 'param', 'index': ALL}, 'id')
#     ]
# )
# def test_params(styl, div, values, ids):
#     if not div: raise PreventUpdate
#     ic(values)
#     ic(ids)
#     return styl