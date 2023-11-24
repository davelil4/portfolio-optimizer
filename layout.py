import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
import linprog as lp
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def build_portfolio():
    pass

def build_port_figure(symbols):
    port = lp.getEffPort(symbols)
    sdP = port["sdP"]
    muP = port["muP"]
    muf = port["muf"]
    
    sharpesPort = lp.getSharpesPort(symbols, port)
    minVarPort = lp.getMinVarPort(symbols, port)
    
    df = pd.DataFrame({
        "sdP": sdP,
        "muP": muP
    })
    
    fig = px.line(df, x = "sdP", y = "muP",
                #   range_x=[0., 0.0325], range_y=[0.00005,0.0015]
                  )
    
    inds = np.argwhere(muP >= muP[minVarPort["ind"]]).flatten()
    # print(inds)
    
    
    fig.add_traces(
        go.Scatter(
            x = [0, .03], y = [muf, muf + sharpesPort["sharpe"].max()*.03], mode='lines', line=dict(color='red'), name="Optimal Portfolios"
        )
    )
    
    fig.add_traces(
        go.Scatter(
            x = sdP[inds], y = muP[inds], line_shape='spline', name="Efficient Frontier", line=dict(color='purple'), hovertemplate='sdP=%{x}<br>muP=%{y}',
        )
    )
    
    fig.add_traces(
        go.Scatter(
            x = [sdP[sharpesPort["ind"]]], y = [muP[sharpesPort["ind"]]], mode="markers", name="Sharpes Portfolio", hovertemplate='sdP=%{x}<br>muP=%{y}'
        )
    )
    
    fig.add_traces(
        go.Scatter(
            x = [sdP[minVarPort["ind"]]], y = [muP[minVarPort["ind"]]], mode="markers", name="Minimum Variance Portfolio", hovertemplate='sdP=%{x}<br>muP=%{y}'
        )
    )
    
    return fig

def build_portfolio_graph_layout():
    pass
    


def build_table():
    html.Div(
        [
            
        ]
    )