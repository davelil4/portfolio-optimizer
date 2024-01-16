import dash_bootstrap_components as dbc
import layout_helper as lay

def drawSimInputs():
    intype = 'number'
    return dbc.Stack([
        lay.createInput("Simulations", "nsims", intype),
        lay.createInput("Days", "ndays", intype),
    ], direction="horizontal", gap=4)