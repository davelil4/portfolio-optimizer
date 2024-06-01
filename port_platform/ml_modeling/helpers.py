import pandas as pd
import inspect
import data_grab as dg
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

models = {
        'RandomForestClassifier': RandomForestClassifier,
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis,
        'LogisticRegression': LogisticRegression,
        'KNeighborsClassifier': KNeighborsClassifier,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'GaussianNB': GaussianNB,
        'ExtraTreesClassifier': ExtraTreesClassifier,
        'AdaBoostClassifier': AdaBoostClassifier,
        'SVM': svm.SVC,
        'GradientBoostingClassifier': GradientBoostingClassifier,
        'MLPClassifier': MLPClassifier
}

def get_function_arguments(func):
    signature = inspect.signature(func)
    arguments = []
    for name, param in signature.parameters.items():
        if param.default is inspect.Parameter.empty:
            arguments.append((name, None))
        else:
            arguments.append((name, param.default))
    return arguments

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
    hist['Date'] = pd.to_datetime(hist['Date'])
    hist.set_index('Date', inplace=True)
    
    return hist

def get_hist(data, symbol):
    if data is None or (pd.to_datetime(data['last_date']).date() < pd.Timestamp.today('America/New_York').date()) or symbol not in data or 'last_date' not in data or symbol not in data:
        hist = dg.getHistory(symbol, 'max')
        data = create_data_from_df(hist, symbol)
    else:
        hist = create_df_from_data(data, symbol)
    
    return hist, data