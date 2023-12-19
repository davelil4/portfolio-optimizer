import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
import linprog as lp
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import data_grab as dg

def build_portfolio():
    pass

def build_port_figure(symbols, port):
    sdP = port["sdP"]
    muP = port["muP"]
    muf = port["muf"]
    
    sharpesPort = lp.getSharpesPort(symbols, port)
    minVarPort = lp.getMinVarPort(symbols, port)
    
    df = pd.DataFrame({
        "sdP": sdP,
        "muP": muP
    })
    
    minMuP = muP.min()
    maxMuP = muP.max()
    
    fig = px.line(df, x = "sdP", y = "muP", title="Reward/Risk for Portfolio",
                  range_x=[0., 1.5*sdP.max()], range_y=[minMuP * .5,maxMuP * 1.5]
                  )
    
    inds = np.argwhere(muP >= muP[minVarPort["ind"]]).flatten()
    # print(inds)
    
    
    fig.add_traces(
        [
            go.Scatter(
                x = [0, .03], y = [muf, muf + sharpesPort["sharpe"].max()*.03], mode='lines', line=dict(color='red'), name="Optimal Portfolios"
            ),
            go.Scatter(
                x = sdP[inds], y = muP[inds], line_shape='spline', name="Efficient Frontier", line=dict(color='purple'), hovertemplate='sdP=%{x}<br>muP=%{y}',
            ),
            go.Scatter(
                x = [sdP[sharpesPort["ind"]]], y = [muP[sharpesPort["ind"]]], mode="markers", name="Sharpes Portfolio", hovertemplate='sdP=%{x}<br>muP=%{y}'
            ),
            go.Scatter(
                x = [sdP[minVarPort["ind"]]], y = [muP[minVarPort["ind"]]], mode="markers", name="Minimum Variance Portfolio", hovertemplate='sdP=%{x}<br>muP=%{y}'
            )
        ]
    )
    
    return fig
    


def build_port_table(symbols, port, budg):
    
    sharpesPort = lp.getSharpesPort(symbols, port)
    minVarPort = lp.getMinVarPort(symbols, port)
    
    costs = np.array(list(map(lambda x: dg.getMarketPrice(x), symbols)))
    
    def numpy_to_formatted(arr):
        formatted_list=list(map(lambda x: format(x, ".3f"), sharpesPort["weights"].tolist()))
        return formatted_list
    
    
    
    df = pd.DataFrame({
        "Stock Symbol": symbols,
        "Sharpe Weight": numpy_to_formatted(sharpesPort["weights"]),
        "Minimum Variance Weight": numpy_to_formatted(minVarPort["weights"]),
        "Sharpe Weight Allocation": ((budg * sharpesPort["weights"]) / costs).astype(int),
        "Min Var Allocation": ((budg * minVarPort["weights"]) / costs).astype(int)
    })
    
    return html.Div(
        [
            dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)
        ]
    )