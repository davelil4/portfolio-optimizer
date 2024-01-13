#!/usr/bin/env python
# coding: utf-8

# ### Regular Initialization

import yfinance as yf
import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score
# from datetime import timedelta
import numpy as np
import plotly.graph_objs as go
# import plotly.express as px
from plotly.offline import iplot, init_notebook_mode
from plotly.subplots import make_subplots
import pandas_market_calendars as mcal
import time as t
import ta
load_dotenv()


def time_func(func):
    def wrapper(*args, **kwargs):
        t1 = t.time()
        res = func(*args, **kwargs)
        t2 = t.time()
        print(func.__name__,":", t2 - t1)
        return res
    return wrapper


def grab_ticker_data(symbol, time):
    DATA_PATH = f"../../data/{symbol}_data_{time}.json"
    if os.path.exists(DATA_PATH):
        # Read from file if we've already downloaded the data.
        with open(DATA_PATH) as f:
            ticker_hist = pd.read_json(DATA_PATH)

    else:
        ticker = yf.Ticker(symbol)
        ticker_hist = ticker.history(period=time)

        # Save file to json in case we need it later.  This prevents us from having to re-download it every time.
        ticker_hist.to_json(DATA_PATH)
    return ticker_hist


def create_shifted_data(ticker_hist, days=1):
    data = ticker_hist.copy()
    del data["Stock Splits"]
    del data["Dividends"]
    data["Target"] = (data["Close"].shift(-days) > data["Close"]).astype(int)
    return data

def generate_preds(train, test, model, predictors, probability=True):

    if train is not None:
        model.fit(train[predictors], train["Target"])
    

    if probability:
        preds = model.predict_proba(test[predictors])[:,1] # second column is prob that price goes up
        preds = pd.Series(preds, index=test.index)
        preds[preds > .6] = 1
        preds[preds <= .6] = 0
    else:
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
    
    # Combine predictions and test values
    combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)
    return combined


def backtest(data, model, predictors, start=1000, step=750, probability=True):
    predictions = []
    # Loop over the dataset in increments
    for i in range(start, data.shape[0], step):
        # Split into train and test sets
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()

        combined = generate_preds(train, test, model, predictors, probability)

        predictions.append(combined)

    return pd.concat(predictions)


# full_predictors = predictors + ["weekly_mean", "quarterly_mean", "annual_mean", "annual_weekly_mean", "annual_quarterly_mean", 
#                                 "open_close_ratio", "high_close_ratio", "low_close_ratio", "weekly_trend", "MACD"]

full_predictors = ["weekly_mean", "quarterly_mean", "annual_mean", "annual_weekly_mean", "annual_quarterly_mean", 
                                "open_close_ratio", "high_close_ratio", "low_close_ratio", "weekly_trend", "MACD"]

def weekly_mean(data):
    data["weekly_mean"] = data.rolling(7).mean()["Close"] / data["Close"]
    return data

def quarterly_mean(data):
    data["quarterly_mean"] = data.rolling(90).mean()["Close"] / data["Close"]
    return data

def annual_mean(data):
    data["annual_mean"] = data.rolling(365).mean()["Close"] / data["Close"]
    return data

def weekly_trend(data):
    data["weekly_trend"] = data.shift(1).rolling(7).sum()["Target"]
    return data

def annual_weekly_mean(data):
    data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
    return data

def annual_quarterly_mean(data):
    data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]
    return data

def open_close_ratio(data):
    data["open_close_ratio"] = data["Open"] / data["Close"]
    return data

def high_close_ratio(data):
    data["high_close_ratio"] = data["High"] / data["Close"]
    return data

def low_close_ratio(data):
    data["low_close_ratio"] = data["Low"] / data["Close"]
    return data

def MACD(data):
    data["MACD"] = ta.trend.ema_indicator(data["Close"], window=26) - ta.trend.ema_indicator(data["Close"], window=12)
    return data

def EMA_ratio(data, wind1, wind2):
    data[f"EMA_{wind1}_{wind2}"] = ta.trend.ema_indicator(data["Close"], window=wind1) / ta.trend.ema_indicator(data["Close"], window=wind2)

pred_to_func = {
    'weekly_mean': weekly_mean,
    'quarterly_mean': quarterly_mean,
    'annual_mean': annual_mean,
    'weekly_trend': weekly_trend,
    'annual_weekly_mean': annual_weekly_mean,
    'annual_quarterly_mean': annual_quarterly_mean,
    'open_close_ratio': open_close_ratio,
    'high_close_ratio': high_close_ratio,
    'low_close_ratio': low_close_ratio,
    'MACD': MACD,
    'EMA_ratio': EMA_ratio
}

# def create_new_predictors(data):
#     weekly_mean = data.rolling(7).mean()["Close"]
#     quarterly_mean = data.rolling(90).mean()["Close"]
#     annual_mean = data.rolling(365).mean()["Close"]

#     weekly_trend = data.shift(1).rolling(7).sum()["Target"]

#     data["weekly_mean"] = weekly_mean / data["Close"]
#     data["quarterly_mean"] = quarterly_mean / data["Close"]
#     data["annual_mean"] = annual_mean / data["Close"]

#     data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
#     data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]

#     data["weekly_trend"] = weekly_trend

#     data["open_close_ratio"] = data["Open"] / data["Close"]
#     data["high_close_ratio"] = data["High"] / data["Close"]
#     data["low_close_ratio"] = data["Low"] / data["Close"]

#     data["MACD"] = ta.trend.ema_indicator(data["Close"], window=26) - ta.trend.ema_indicator(data["Close"], window=12)


#     return data

def create_new_predictors(data, predictors):
    for pred in predictors:
        if pred in pred_to_func:
            data = pred_to_func[pred](data)
    return data

def generate_dates(calendar, last_date, days, json=True):
    if json: 
        return (calendar.valid_days(last_date, end_date=last_date + pd.Timedelta(days, "d"), tz="America/New_York")[1:] + pd.Timedelta(5, "h")).tz_localize(None)
    return (calendar.valid_days(last_date, end_date=last_date + pd.Timedelta(days, "d"), tz="America/New_York")[1:])


def generate_intraday_returns(s_mu, s_sigma, steps_per_day, days, sig_scal=1, v_scal=1):

    s_sigma = sig_scal * s_sigma * np.sqrt(252)

    s_mu = ((s_mu + 1)**(252)) - 1

    T = days / 252.0  # Business days in a year
    # T = 1
    s_dt = T / (steps_per_day * days)  # 4.0 is needed as four prices per day are required

    total_steps = steps_per_day * days
    returns = np.exp((s_mu - s_sigma**2 / 2) * s_dt + s_sigma * np.random.normal(0, np.sqrt(s_dt), size=total_steps))

    return returns


def generate_intraday_prices(initial_price, s_mu, s_sigma, steps_per_day, days, dates, sig_scal=1, v_scal=1):
    returns = generate_intraday_returns(s_mu, s_sigma, steps_per_day, days, sig_scal, v_scal)
    
    close_prices = initial_price * np.cumprod(returns, axis=0)
    
    # Reshape the close prices to steps_per_day rows and days columns
    close_prices_reshaped = close_prices.reshape((days, steps_per_day))
    
    # Extract open, high, low, and close prices from intraday data
    open_prices = close_prices_reshaped[:, 0]
    close_prices = close_prices_reshaped[:, -1]
    high_prices = np.max(close_prices_reshaped, axis=1)
    low_prices = np.min(close_prices_reshaped, axis=1)
    

    # ind_array = np.array([last_date + timedelta(days=i) for i in range(1, days+1)]) # Switch to method that doesn't include weekends in future

    # Create a DataFrame to store the prices
    price_data = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
    }, index = dates)
    
    return price_data


def forecast_GBM(data, steps_per_day, days, dates, sig_scal=1, v_scal=1):
    close = data["Close"]
    close_ret = ((close / close.shift(1)) - 1).iloc[1:]
    # volume = data["Volume"]
    # volume_ret = ((volume / volume.shift(1)) - 1).iloc[1:]
    s_mu = close_ret.mean()
    s_sigma = close_ret.std()
    # print(s_mu, s_sigma)
    # v_mu = volume_ret.mean()
    # v_sigma = volume_ret.std()
    init_price = close.iloc[-1]
    # init_vol = volume.iloc[-1]
    d = generate_intraday_prices(init_price,
                                #  init_vol, 
                                 s_mu, s_sigma, 
                                #  v_mu, v_sigma, 
                                 steps_per_day, days, dates, sig_scal, 
                                #  v_scal
                                 )
    
    # d["GRet"] = d["Close"].pct_change(1)
    # d["NRet"] = d["Close"] / d["Close"].shift(1)

    # d["Volume"] = np.random.choice(volume.iloc[-days:], size=days, replace=True)
    return d


calendar = mcal.get_calendar('NYSE')


def runSims(hist, model, full_predictors, n_days, n_sims, sig_scal=1, v_scal=1):
    sims = []
    preds = []
    all_rets = np.zeros(n_sims)
    # profits = np.zeros(n_sims)

    calendar = mcal.get_calendar('NYSE')
    last_date = hist.iloc[[-1]].index[0]
    # print(hist)
    dates = generate_dates(calendar, last_date, n_days, json=False)
    tdays = len(dates)
    
    closes = np.zeros((n_sims, tdays + 1))
    all_preds = np.zeros((n_sims, tdays))
    
    start = -(min(len(hist) - 1, 5 * n_days))
    start = 365 if len(hist) + start < 365 else start

    train = create_new_predictors(create_shifted_data(hist, 1), full_predictors).iloc[start:-1]
    model.fit(train[full_predictors], train["Target"])
    
    for i in range(n_sims):
        
        forecast = forecast_GBM(hist, 4, tdays, dates, sig_scal, v_scal)
        
        test = create_new_predictors(create_shifted_data(pd.concat([hist, forecast]), 1), full_predictors)[-(len(forecast)):]
        inst_preds = generate_preds(None, test, model, full_predictors, probability=True)

        forecast = pd.concat([hist.iloc[[-1]], forecast])

        sims.append(forecast)

        preds.append(inst_preds)
        
        closes[i, :] = forecast["Close"].to_numpy()
        
        all_preds[i, :] = inst_preds["Predictions"].to_numpy()

    grets = ((np.roll(closes, -1, axis=1) - closes) / (np.roll(closes, -1, axis=1)))[:, :-1]
    
    fin_preds = all_preds
    
    profit = fin_preds * (closes - np.roll(closes, 1, axis=1))[:, 1:]
    
    preds = pd.concat(preds)
    
    grets_avg = np.sum(fin_preds * grets, axis=1).mean()
    profit = np.sum(profit, axis=1).mean()

    return {
        "data" : sims,
        "preds" : preds,
        "rets": all_rets.mean(),
        "profit": profit,
        # "nrets_tot": nrets,
        "grets": np.cumsum(grets, axis=1),
        "grets_avg": grets_avg,
        "strat_grets": np.cumsum(grets * fin_preds, axis=1),
        # "finpreds": fin_preds,
        "precision": precision_score(preds["Target"], preds["Predictions"])
        }


# ### Monte Carlo Plots

def simFigure(ticker_hist, sims):

    # create list of the line segments you want to plot
    all_xs =[sims[0].index.copy() for _ in range(len(sims))]
    all_ys = list(map(lambda x: x["Close"], sims))
    # print(all_ys)

    # all_vys = list(map(lambda x: x["Volume"], sims))
    # print(all_vys)

    # append nan to each segment
    all_xs_with_nan = [np.concatenate((xs, [np.datetime64("NaT")])) for xs in all_xs]
    all_ys_with_nan = [np.concatenate((ys, [np.nan])) for ys in all_ys]
    # all_vys_with_nan = [np.concatenate((ys, [np.nan])) for ys in all_vys]



    # concatinate segments into single line
    xs = np.concatenate(all_xs_with_nan)
    ys = np.concatenate(all_ys_with_nan)
    # vys = np.concatenate(all_vys_with_nan)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Close", "Volume"))

    fig.add_trace(
        go.Scattergl(x=xs, y=ys, mode='lines', opacity=.05, line={'color': 'darkblue'}, name="Stock Price Sim"), 1, 1
    )

    # fig.add_trace(
    #     go.Scattergl(x=xs, y=vys, mode='lines', opacity=.05, line={'color': 'darkred'}, name="Volume Sim"), 1, 2
    # )

    lim = 365

    fig.add_trace(
        go.Scatter(x=ticker_hist.tail(lim).index, y=ticker_hist["Close"].tail(lim), name="Stock Price Hist"), 1, 1
    )
    # fig.add_trace(
    #     go.Scatter(x=ticker_hist.tail(lim).index, y=ticker_hist["Volume"].tail(lim), name = "Volume Hist"), 1, 2
    # )

    return fig

def stratFigure(ticker_hist, res, nsims, ndays):
    # ### Simulations Vs. Strategy

    last_date = ticker_hist.iloc[[-1]].index[0]
    dates = generate_dates(calendar, last_date, ndays)


    # create list of the line segments you want to plot
    all_xs = [dates for _ in range(nsims)]
    all_ys = res["grets"]
    # print(all_ys)

    all_strat_ys = res["strat_grets"]
    # print(all_vys)

    # print(len(dates), res["strat_grets"].shape, res["grets"].shape)


    # append nan to each segment
    all_xs_with_nan = [np.concatenate((xs, [np.datetime64("NaT")])) for xs in all_xs]
    all_ys_with_nan = [np.concatenate((ys, [np.nan])) for ys in all_ys]
    all_strat_ys_with_nan = [np.concatenate((ys, [np.nan])) for ys in all_strat_ys]



    # concatinate segments into single line
    xs = np.concatenate(all_xs_with_nan)
    ys = np.concatenate(all_ys_with_nan)
    strat_ys = np.concatenate(all_strat_ys_with_nan)

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(x=xs, y=ys, mode='lines', opacity=.1, line={'color': 'darkblue'}, name="Stock")
    )

    fig.add_trace(
        go.Scattergl(x=xs, y=strat_ys, mode='lines', opacity=.2, line={'color': 'darkred'}, name="Strategy")
    )

    return fig