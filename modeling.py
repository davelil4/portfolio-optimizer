import yfinance as yf
import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import numpy as np
load_dotenv()

def predictStock(symbol):
    DATA_PATH = f"data/{symbol}_data.json"
    if os.path.exists(DATA_PATH):
        # Read from file if we've already downloaded the data.
        with open(DATA_PATH) as f:
            ticker_hist = pd.read_json(DATA_PATH)
    
    else:
        ticker = yf.Ticker(symbol)
        ticker_hist = ticker.history(period="max")

        # Save file to json in case we need it later.  This prevents us from having to re-download it every time.
        ticker_hist.to_json(DATA_PATH)
    # Ensure we know the actual closing price
    data = ticker_hist[["Close"]]
    data = data.rename(columns = {'Close':'Actual_Close'})

    # Setup our target.  This identifies if the price went up or down
    data["Target"] = ticker_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]
    
    # Shift stock prices forward one day, so we're predicting tomorrow's stock prices from today's prices.
    ticker_prev = ticker_hist.copy()
    ticker_prev = ticker_prev.shift(1)
    
    predictors = ["Close", "Volume", "Open", "High", "Low"]
    data = data.join(ticker_prev[predictors]).iloc[1:]
    model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)
    # train = data.iloc[:-100]
    # test = data.iloc[-100:]

    # model.fit(train[predictors], train["Target"])
    # preds = model.predict(test[predictors])
    # preds = pd.Series(preds, index=test.index)
    # combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)
    # i = 1000
    # step = 750

    # train = data.iloc[0:i].copy()
    # test = data.iloc[i:(i+step)].copy()
    # model.fit(train[predictors], train["Target"])
    # preds = model.predict(test[predictors])
    
    # preds = model.predict_proba(test[predictors])[:,1]
    # preds = pd.Series(preds, index=test.index)
    # preds[preds > .6] = 1
    # preds[preds<=.6] = 0
    
    # predictions = []
    # # Loop over the dataset in increments
    # for i in range(1000, data.shape[0], step):
    #     # Split into train and test sets
    #     train = data.iloc[0:i].copy()
    #     test = data.iloc[i:(i+step)].copy()

    #     # Fit the random forest model
    #     model.fit(train[predictors], train["Target"])

    #     # Make predictions
    #     preds = model.predict_proba(test[predictors])[:,1]
    #     preds = pd.Series(preds, index=test.index)
    #     preds[preds > .6] = 1
    #     preds[preds<=.6] = 0

    #     # Combine predictions and test values
    #     combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)

    #     predictions.append(combined)
        
    def backtest(data, model, predictors, start=1000, step=750):
        predictions = []
        # Loop over the dataset in increments
        for i in range(start, data.shape[0], step):
            # Split into train and test sets
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i+step)].copy()

            # Fit the random forest model
            model.fit(train[predictors], train["Target"])

            # Make predictions
            preds = model.predict_proba(test[predictors])[:,1]
            preds = pd.Series(preds, index=test.index)
            preds[preds > .6] = 1
            preds[preds<=.6] = 0

            # Combine predictions and test values
            combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)

            predictions.append(combined)

        return pd.concat(predictions)
    
    # predictions = backtest(data, model, predictors)
    
    weekly_mean = data.rolling(7).mean()["Close"]
    quarterly_mean = data.rolling(90).mean()["Close"]
    annual_mean = data.rolling(365).mean()["Close"]
    
    weekly_trend = data.shift(1).rolling(7).sum()["Target"]
    
    data["weekly_mean"] = weekly_mean / data["Close"]
    data["quarterly_mean"] = quarterly_mean / data["Close"]
    data["annual_mean"] = annual_mean / data["Close"]
    
    data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
    data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]
    
    data["weekly_trend"] = weekly_trend
    
    data["open_close_ratio"] = data["Open"] / data["Close"]
    data["high_close_ratio"] = data["High"] / data["Close"]
    data["low_close_ratio"] = data["Low"] / data["Close"]
    
    full_predictors = predictors + ["weekly_mean", "quarterly_mean", "annual_mean", "annual_weekly_mean", "annual_quarterly_mean", "open_close_ratio", "high_close_ratio", "low_close_ratio"]

    predictions = backtest(data.iloc[365:], model, full_predictors)
    
    return predictions.plot(backend='plotly')



def forwardtest(data, model, predictors, n_days, n_sims, symbol):
    sims = []
    
    for sim in n_sims:
        predictions = []

        d = data[-100:]
        
        for step in range(100):
            train = d
            
            test_dict = {}
            
            for col, _ in d.item():
                test_dict[col] = np.random.choice(d[col], size=100, replace=True)
            
            test = pd.DataFrame(test_dict)
            
            model.fit(train[predictors], train["Target"])
            
            preds = model.predict_proba(test[predictors])[:,1]
            preds = pd.Series(preds, index=test.index)
            preds[preds > .6] = 1
            preds[preds<=.6] = 0
            
            # Combine predictions and test values
            combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)

            predictions.append(combined)
            