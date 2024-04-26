from dash import html, Output, Input, callback, State
from dash.exceptions import PreventUpdate
import dash.dcc as dcc
import dash_bootstrap_components as dbc
import plotly.express as px
from ml_modeling.helpers import create_df_from_data

def make_layout():
    return html.Div([
            dcc.Graph('ind-graph'),
            html.Br(),
            dcc.Checklist(id='ind-checklist', options=[], value=[], inline=True)
        ], id="ind_comps")
    

def make_callbacks():
    
    @callback(
        Output('ind-checklist', 'options'),
        # Output('ind-checklist', 'value'), Change values to only whats in the columns
        Input('indicator-df', 'data'),
        State('dd_ms', 'value')
    )
    def update_checklist(data, symbol):
        if not data:
            raise PreventUpdate
        df = create_df_from_data(data, symbol)
        return df.columns
    
    @callback(
        Output('ind-graph', 'figure'),
        State('indicator-df', 'data'),
        State('dd_ms', 'value'),
        Input('ind-checklist', 'value')
    )
    def update_ind_graph(data, symbol, inds):
        if not data:
            raise PreventUpdate
        return px.line(create_df_from_data(data, symbol)[inds])
    
    