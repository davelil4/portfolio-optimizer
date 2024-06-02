from dash import html, dcc, callback, Input, Output, State, ALL, MATCH
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from ml_modeling.helpers import *
import joblib as jb
import pandas as pd


def make_create_layout():
    return html.Div([
        html.Div([
            html.H5('Model Creation'),
            dbc.Stack([
                html.Div(dcc.Dropdown(list(models.keys()), id='model_select'), style={'width':'50%', 'height':'100%'}),
                dbc.Input(id='model_name', type='text', placeholder='Model Name', style={'width':'50%', 'height':'100%'})
            ], direction='horizontal', gap=3),
            html.Div(id='model_params')
            ]),
            html.Br(),
            dbc.Stack([
                dbc.Button('Save Model', 'b_save', disabled=True),
                dbc.Label(id='l_succ')
        ], direction='horizontal', gap=3)
    ], style={'height':'100%'})

def make_select_layout():
    return html.Div([
        html.H5('Model Selection'),
        html.Div(id='model_btns'),
        html.Br(),
        dbc.Collapse(id='model_collapse', is_open=False)
    ])

def model_from_inputs(model_name, input_dicts, values):
    params = list(map(lambda x: x['index'], input_dicts))
    param_map = {params[i]: values[i] for i in range(len(values))}
    return models[model_name](**param_map)

def create_table(params, values):
    return dbc.Table.from_dataframe(
        pd.DataFrame(
            {
                'Params': params,
                'Values': values
            }
        ), striped=True, bordered=True, hover=True
    )

def make_callbacks():
    @callback(
        [
            Output('model_params', 'children'),
            Output('b_save', 'disabled')
        ],
        Input('model_select', 'value')
    )
    def create_model_params(model_name):
        
        def create_param(arg, default):
            return dbc.Stack(
                    [
                        dbc.Label(f'{arg}: '),
                        dbc.Input({'type': 'model_param', 'index': arg}, value=default)
                    ], direction='horizontal', gap=3
                )
        
        if model_name is None:
            return [], True
        
        
        model = models[model_name]
        
        params = []
        
        for arg, default in get_function_arguments(model):
            params.append(create_param(arg, default))
        
        return params, False
    
    @callback(
        [
            Output('models', 'data'),
            Output('l_succ', 'children')
        ],
        [
            Input('b_save', 'n_clicks'),
            State('model_select', 'value'),
            State({'type': 'model_param', 'index': ALL}, 'value'),
            State({'type': 'model_param', 'index': ALL}, 'id'),
            State('model_name', 'value'),
            State('models', 'data')
        ],
        background=True,
        running=[
            (Output('b_save', 'disabled'), True, False),
        ],
        prevent_initial_call=True
    )
    def save_model(b_save, model_type, vals, ids, model_name, models):
        if b_save is None:
            raise PreventUpdate
        
        if models is None:
            models = {}
        
        if model_type is None:
            return models, 'No model selected.'
        elif model_name is None:
            return models, 'No model name.'
        
        params = list(map(lambda x: x['index'], ids))
        
        models[model_name] = dict(zip(params, vals))
        
        models[model_name]['model'] = model_type
        
        jb.dump(model_from_inputs(model_type, ids, vals), f'{model_name}.joblib')
        
        return models, 'Successfully saved model.'

    
    @callback(
        Output('model_btns', 'children'),
        Input('models', 'data'),
    )
    def add_model_button(models):
        if models is None:
            raise PreventUpdate
        
        return [dbc.Button(name, id={'type': 'model_btn', 'index': f'{name}'}) for name in models.keys()]
    
    @callback(
        [
            Output({'type': 'model_btn', 'index': ALL}, 'disabled'),
            Output('model_collapse', 'is_open'),
            Output('model_collapse', 'children')
        ],
        Input({'type': 'model_btn', 'index': ALL}, 'n_clicks_timestamp'),
        State('models', 'data'),
        State({'type': 'model_btn', 'index': ALL}, 'id')
    )
    def disable_model_btns(ts, models, btn_ids):
        if ts is None or len(ts) == 0:
            raise PreventUpdate
        
        ts = [x if x is not None else 0 for x in ts]
        
        idx = ts.index(max(ts))
        
        res = [False for _ in ts]
        res[idx] = True
        
        model = btn_ids[idx]['index']
        
        del models[model]['model']
        
        return res, True, [create_table(list(models[model].keys()), list(models[model].values()))]
    
    @callback(
        Output('curr_model', 'data'),
        Input({'type': 'model_btn', 'index': ALL}, 'n_clicks_timestamp'),
        State('models', 'data'),
        State({'type': 'model_btn', 'index': ALL}, 'id')
    )
    def set_curr_model(ts, models, btn_ids):
        ts = [x if x is not None else 0 for x in ts]
        
        idx = ts.index(max(ts))
        
        res = [False for _ in ts]
        res[idx] = True
        
        model = btn_ids[idx]['index']
        
        return models[model]
        
        