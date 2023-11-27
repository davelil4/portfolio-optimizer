import data_grab as dg
import pandas as pd
from scipy.optimize import linprog
import numpy as np
from qpsolvers import solve_qp
import math
import pyreadr

tempMuf = 0.02/253

syms = ["AAPL", "AEP", "BAX", "ED", "F", "GE", "GOOG", "MCD", "MSFT"]

def getEffPort(symbols, lowerb=-.15, upperb=.5):
    # print(lowerb)
    lb = lowerb
    ub = upperb
    
    muf = tempMuf
    
    # Finding Bounds with LP
    
    m = len(symbols)
    
    rets = dg.getLogReturnsFromList(symbols)
    AmatLP1 = np.column_stack((np.diag(np.diag(np.full((m, m),1))), np.full((m, m),0)))
    AmatLP2 = np.column_stack((np.full((m, m),0), np.diag(np.diag(np.full((m, m),1)))))
    # AmatLP3 = np.column_stack((np.full((1, m), 1), np.full((1, m), -1)))
    AmatUB = np.row_stack((AmatLP1, AmatLP2))
    # AmatUB = np.diag(np.diag(np.full((m, m), 1)))

    AmatEQ = np.column_stack((np.full((1, m), 1), np.full((1, m), -1)))
    # AmatEQ = np.full((1, m), 1)
    
    # res = pyreadr.read_r('returns.Jul25.2023.RData')
    # print(res['returns']["AAPL"])
    # rets = pd.DataFrame(res["returns"])[syms]
    
    means = rets.mean()
    # print(means.to_numpy())
    # sds = rets.std()
    
    bounds = (lb, ub)
    
    B_ub = np.column_stack((np.full((1, m), ub), np.full((1, m), -1 * lb)))
    # B_ub = np.full((1, m), ub)
    B_eq = np.ones(1)
    
    cvec=np.column_stack((means.to_numpy().reshape((1, m)), -1 * means.to_numpy().reshape((1, m))))
    # cvec = means.to_numpy().reshape((1, m))
    
    # print(AmatUB.shape, AmatEQ.shape, cvec.shape, B_ub.shape, B_eq.shape)
    
    res_max = linprog(c=(-1 * cvec), A_ub=AmatUB, A_eq=AmatEQ, b_ub=B_ub, b_eq=B_eq)
    res_min = linprog(c=cvec, A_ub=AmatUB, A_eq=AmatEQ, b_ub=B_ub, b_eq=B_eq)
    mu_bounds = [res_min.get("fun"), -1 * res_max.get("fun")]
    
    # print(mu_bounds)
    
    # Using QP for Min Var and Rest
    
    Dmat = 2 * np.cov(rets, rowvar=False)
    dvec = np.full((1, m), 0.)
    eps = .000000001
    
    muP = np.linspace(start=mu_bounds[0] + eps, stop=mu_bounds[1] - eps, num=300)
    sdP = muP.copy()
    weights = np.full((300,m), 0.)
    
    Amat = np.column_stack((
        np.full((m,1), 1), 
        means.to_numpy().reshape((m, )), 
    ))
    
    # print(Amat)
    
    hvec = np.row_stack((
        np.full((m, 1), (lb)), np.full((m, 1), -ub)
    ))
    
    Gmat = np.column_stack((
        # np.full((m,1), 1), 
        # means.to_numpy().reshape((m, )), 
        np.diag(np.diag(np.full((m,m), 1))), np.diag(np.diag(np.full((m,m), -1)))
        ))
    
    # print(Gmat)
    
    for i in range(300):
        bvec = np.row_stack((np.ones(1), np.full((1, 1), muP[i]), 
                            #  np.full((m, 1), (lb)), np.full((m, 1), -ub)
                             ))
        res = solve_qp(P=Dmat, q=(-1 * dvec), G=(-1 * Gmat.transpose()), h=(-1 * hvec), 
                       A=Amat.transpose(), b=bvec,
                    #    lb=np.full((1, m), (lb)), ub=np.full((1, m), ub), 
                       solver='quadprog')
        
        sdP[i] = math.sqrt( ((0.5 * res.T @ Dmat @ res) - (dvec @ res))[0] )
        weights[i, :] = res
    return {
        "muP": muP,
        "sdP": sdP,
        "weights": weights,
        "muf": muf
    }

def getSharpesPort(symbols, init_port=None):
    port = init_port if init_port else getEffPort(symbols)
    muP = port["muP"]
    sdP = port["sdP"]
    muf = port["muf"]
    weights = port["weights"]
    sharpes = (muP - muf) / sdP
    ind = np.argmax(sharpes)
    
    return {
        "weights": weights[ind, :],
        "ind": ind,
        "sharpe": sharpes
    }

def getMinVarPort(symbols, init_port=None):
    port = init_port if init_port else getEffPort(symbols)
    sdP = port["sdP"]
    weights = port["weights"]
    ind = np.argmin(sdP)
    return {
        "weights": weights[ind, :],
        "ind": ind
    }