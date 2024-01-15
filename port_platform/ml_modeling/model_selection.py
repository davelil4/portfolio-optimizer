from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from ml_modeling.modeling import backtest
import plotly.express as px
import pandas as pd

def model_selection(train, predictors):
    seed = 8
    models = []
    models.append(('LogisticRegression', LogisticRegression(random_state=seed), False))
    models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis(), False))
    models.append(('KNeighborsClassifier', KNeighborsClassifier(), False))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier(), False))
    models.append(('GaussianNB', GaussianNB(), False))
    models.append(('RandomForestClassifier', RandomForestClassifier(), True))
    models.append(('ExtraTreesClassifier',ExtraTreesClassifier(random_state=seed), False))
    models.append(('AdaBoostClassifier',AdaBoostClassifier(DecisionTreeClassifier(random_state=seed),random_state=seed,learning_rate=0.1), False))
    models.append(('SVM',svm.SVC(random_state=seed), False))
    models.append(('GradientBoostingClassifier',GradientBoostingClassifier(random_state=seed), False))
    models.append(('MLPClassifier',MLPClassifier(random_state=seed), False))
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'
    for name, model, prob in models:
        # kfold = KFold(n_splits=10, shuffle=True, random_state=seed) 
        preds = backtest(train, model, predictors, probability=prob)
        cv_results = precision_score(preds["Target"], preds["Predictions"])
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results, cv_results.std())
        print(msg) 
    return results, names

def drawMSFigure(results, names):

    resultdf = pd.DataFrame(data={
        'name': names,
        'score': results
    })

    # Model with best score
    # print(resultdf["name"].iloc[resultdf["score"].idxmax()])

    fig = px.bar(resultdf, x='score', y='name', orientation='h')
    
    return fig

def getBestModel(results, names):
    return names[pd.Series(results).idxmax()]
    
    