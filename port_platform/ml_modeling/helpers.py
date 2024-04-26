import pandas as pd


def create_data_from_df(df, symbol):
    data = {}
    hist = df
    hist['Date'] = hist.index
    data[symbol] = hist.to_dict('records')
    del hist['Date']
    data['last_date'] = hist.iloc[[-1]].index[0]
    
    return data

def create_df_from_data(data, symbol):
    hist = pd.DataFrame.from_dict(data[symbol])
    hist.set_index('Date')
    del hist['Date']
    
    return hist