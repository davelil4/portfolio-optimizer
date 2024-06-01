from dash import html, dcc, callback, Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from ml_modeling.helpers import *
import joblib as jb


def make_layout():
    return html.Div([
        html.Div([
            html.H5('Model'),
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

def model_from_inputs(model_name, input_dicts, values):
    params = list(map(lambda x: x['index'], input_dicts))
    param_map = {params[i]: values[i] for i in range(len(values))}
    return models[model_name](**param_map)

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
        
        models[model_name] = dict(zip(ids, vals))
        
        jb.dump(model_from_inputs(model_type, ids, vals), f'{model_name}.joblib')
        
        return models, 'Successfully saved model.'

    