from dash import html, dcc
import dash_bootstrap_components as dbc

def createInput(name, id, intype):
    return dbc.Stack([
        dbc.Label(name),
        dbc.Input(id, type=intype)
    ], direction='horizontal', gap=3)