import logging    # first of all import the module
from dash import Dash, html, Output, Input, DiskcacheManager
import dash_bootstrap_components as dbc
from portfolio_info.complete import *
from ml_modeling.complete import *
import diskcache
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

logging.basicConfig(filename='std.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.warning('This message will get logged on to a file')

# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')


app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE], background_callback_manager=background_callback_manager)


app.layout = dbc.Container(
    [
        dcc.Store(id='ticker_data', storage_type='local'),
        html.H1(children='Portfolio Optimizer', style={'textAlign':'center'}),
        html.Hr(),
        dbc.Tabs(
            [
                dbc.Tab(label="Portfolio Info", tab_id="info"),
                dbc.Tab(label="Machine Learning", tab_id="ml"),
            ],
            id="tabs",
            active_tab="info",
            persistence=True,
        ),
        html.Br(),
        info_tab,
        ml_tab
    ], fluid=True
)

@app.callback(
    [Output("info_tab", "hidden"), Output("ml_tab", "hidden")],
    [Input("tabs", "active_tab")],
)
def render_tab_content(active_tab):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if active_tab is not None:
        if active_tab == "info":
            return (False, True)
        elif active_tab == "ml":
            return (True, False)
    return (True, False)

if __name__ == '__main__':
    app.run(debug=True)
