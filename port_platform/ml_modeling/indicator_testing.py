from dash import html, Output, Input, callback, State, ALL, dash_table
from dash.exceptions import PreventUpdate
import dash.dcc as dcc
import dash_bootstrap_components as dbc
import ta
from ml_modeling.helpers import get_function_arguments

### TA Generator | Dash Preview


submodules = [name for name in dir(ta) if not (name.startswith("__") or name.startswith("add") or name == 'wrapper' or name == 'utils')]

mod_to_inds = {}
    
for module in submodules:
    function_names = [
        name for name in dir(getattr(ta, module)) 
        if callable(getattr(getattr(ta, module), name)) and 
        not (name.startswith("__") or name.startswith("_") or name[0].isupper())]
    mod_to_inds[module] = function_names

def get_ind_args(module, ind):
    return list(filter(lambda x: (x[0].lower() not in ["close", "open", "high", "low", "volume"]), get_function_arguments(getattr(getattr(ta, module), ind))))

def draw_ind_inputs(module, ind):
    inputs = []
    for arg, val in get_ind_args(module, ind):
        inputs.append(dbc.Stack([
            dbc.Label(arg),
            dbc.Input(id={'type':'ind-param', 'index':arg}, value=val)
        ], direction='horizontal', gap=2))
    return inputs

def make_layout():
    return html.Div([
        dcc.Store(id='ind-store', storage_type='memory', data={'trend': {}, 'momentum': {}, 'volume': {}, 'volatility': {}, 'others': {}}),
        html.H5("Indicators"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Stack([
                            dbc.Label("Indicator Type", style={'width': '30%'}),
                            dcc.Dropdown(submodules, id="submod-dd", style={'width':'70%'}),
                        ], direction='horizontal', gap=2)
                    ]
                ),
                dbc.Col(
                    [
                        dbc.Stack([
                            dbc.Label("Indicator", style={'width': '30%'}),
                            dcc.Dropdown(id = 'ind-dd', style={'width':'70%'})
                        ], direction='horizontal', gap=2)
                    ]
                )
            ]
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col((
                    [
                        html.Div(id='ind-params'),
                        html.Br(),
                        dbc.Button("Add Indicator", id='add-ind', color='primary'),
                        # html.Div(id='test-data')
                    ]
                ))
            ]
        ),
        dbc.Row(
            [
                dbc.Col([
                    dash_table.DataTable(id='ind-table', row_deletable=True, style_cell={'textAlign': 'left'},)
                ])
            ]
        )
    ])

def make_callbacks():

    @callback(
        Output('ind-dd', 'options'),
        Input('submod-dd', 'value')
    )
    def update_ind_dropdown(submod):
        if not submod:
            raise PreventUpdate
        return mod_to_inds[submod]

    @callback(
        Output('add-ind', 'disabled'), 
        Input('ind-dd', 'value'),
        Input('submod-dd', 'value')
    )
    def disable_button(ind, mod):
        if not ind or not mod:
            return True
        return False

    @callback(
        Output('ind-params', 'children'),
        Input('ind-dd', 'value'),
        Input('submod-dd', 'value')
    )
    def update_ind_adder(ind, mod):
        if not ind or not mod:
            return None
        return draw_ind_inputs(mod, ind)

    @callback(   
        [
            Output('ind-store', 'data', allow_duplicate=True),
            Output('ind-table', 'data')
        ],
        [
            Input('add-ind', 'n_clicks'),
            State('ind-store', 'data'),
            State({'type': 'ind-param', 'index': ALL}, 'value'),
            State({'type': 'ind-param', 'index': ALL}, 'id'),
            State('ind-dd', 'value'),
            State('submod-dd', 'value'),
            State('ind-table', 'data')
        ],
        prevent_initial_call='initial_duplicate'
    )
    def add_indicator(clicks, data, vals, ids, ind, mod, tabdata):
        if not clicks or clicks == 0:
            raise PreventUpdate

        if data == None:
            data = {}
        
        args = get_function_arguments(getattr(getattr(ta, mod), ind))
        if ind not in data[mod]:
            data[mod][ind] = {}
        
        for col in ["open", "high", "low", "close"]:
            for arg in args:
                if col == arg[0]:
                    data[mod][ind] = {
                        **data[mod][ind],
                        arg[0]: arg[1],
                    }
                    break
        
        params = {
            ids[i]['index']: vals[i] for i in range(len(ids))
        }
        
        data[mod][ind] = {
            **data[mod][ind],
            **params
        }
        
        if not tabdata:
            tabdata = []

        return data, tabdata + [{'Indicator Type': mod, 'Indicator': ind}]

    @callback(
        Output('ind-store', 'data', allow_duplicate=True),
        Input('ind-table', 'data'),
        State('ind-table', 'data_previous'),
        State('ind-store', 'data'),
        prevent_initial_call='initial_duplicate'
    )
    def remove_data(tdata, old, data):
        if old is None:
            old = []
        else:
            old = list(map(lambda x: (x["Indicator Type"], x['Indicator']), old))
        if tdata is None:
            tdata = []
        else:
            tdata = list(map(lambda x: (x["Indicator Type"], x['Indicator']), tdata))
        
        if len(old) <= len(tdata):
            raise PreventUpdate
        
        
        for mod, ind in old:
            if (mod, ind) not in tdata:
                del data[mod][ind]
        return data


# ### TA Generation | DF

def create_shifted_data(ticker_hist, days=1):
    # Ensure we know the actual closing price
    data = ticker_hist.copy()
    # Setup our target.  This identifies if the price went up or down
    data["Target"] = (data["Close"].shift(-days) > data["Close"]).astype(int)
    # Shift stock prices forward one day, so we're predicting tomorrow's stock prices from today's prices.
    return data

# Need to include type of indicator in the dictionary

def gen_indicators(data, indicators):
    for typ, ind in indicators.items():
        for name, params in ind.items():
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col.lower() in params:
                    params[col.lower()] = data[col]
            
            data[name] = getattr(getattr(ta, typ), name)(**params)
    return data

def get_indicators(inds):
    f = []
    for typ in inds:
        for ind in inds[typ]:
            f.append(ind)
    return f