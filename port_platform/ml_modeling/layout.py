from dash import html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.express as px
# import pandas as pd
import data_grab as dg
import ml_modeling.modeling as ml
import layout_helper as lay




def drawSimInputs():
    intype = 'number'
    return dbc.Stack([
        lay.createInput("Simulations", "nsims", intype),
        lay.createInput("Days", "ndays", intype),
    ], direction="horizontal", gap=4)