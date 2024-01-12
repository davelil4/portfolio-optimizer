from dash import Dash, html, Output, Input
import dash_bootstrap_components as dbc
from portfolio_info.complete import *

# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')


app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])


app.layout = dbc.Container(
    [
        html.H1(children='Portfolio Optimizer', style={'textAlign':'center'}),
        html.Hr(),
        dbc.Tabs(
            [
                dbc.Tab(label="Portfolio Info", tab_id="info"),
                dbc.Tab(label="Machine Learning", tab_id="ml"),
            ],
            id="tabs",
            active_tab="info",
        ),
        info_tab,
    ]
)

@app.callback(
    Output("info_tab", "hidden"),
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
            return False
        elif active_tab == "ml":
            return True
    return "No tab selected"

if __name__ == '__main__':
    app.run(debug=True)
