import data_grab as dg
import pandas as pd
from scipy.optimize import linprog
import numpy as np
from qpsolvers import solve_qp
import math

syms = ["AAPL", "AEP", "BAX", "ED", "F", "GE", "GOOG", "MCD","MSFT"]

def getMinVarPort(symbols):
    
    # Finding Bounds with LP
    
    m = len(symbols)
    
    rets = dg.getLogReturnsFromList(symbols)
    # AmatLP1 = np.column_stack((np.diag(np.full(m,1)), np.full(m,0)))
    # AmatLP2 = np.column_stack((np.full(m,0), np.diag(np.full(m,1))))
    # AmatLP3 = np.column_stack((np.full((1, m), 1), np.full((1, m), -1)))
    # AmatLP = np.row_stack((AmatLP1, AmatLP2, AmatLP3))
    AmatUB = np.diag(np.diag(np.full((m, m), 1)))

    AmatEQ = np.full((1, m), 1)
    
    means = rets.mean()
    sds = rets.std()
    
    bounds = (-.15, .5)
    
    B_ub = np.full((1, m), .5)
    
    cvec=(means.to_numpy().reshape((1, m)))
    
    res_max = linprog(c=(-1 * cvec), bounds=bounds, A_ub=AmatUB, A_eq=AmatEQ, b_ub=B_ub, b_eq=np.ones(1))
    res_min = linprog(c=means, bounds=bounds, A_ub=AmatUB, A_eq=AmatEQ, b_ub=B_ub, b_eq=np.ones(1))
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
        # np.diag(np.diag(np.full((m,m), 1))), np.diag(np.diag(np.full((m,m), -1)))
        ))
    
    print(Dmat.shape, dvec.shape, Amat.shape)
    
    for i in range(300):
        bvec = np.row_stack((np.full((1,1), 1), np.full((1, 1), muP[i]), 
                            #  np.full((m, 1), (-.15)), np.full((m, 1), -.5)
                             ))
        # print(Amat.shape, bvec.shape)
        res = solve_qp(P=Dmat, q=(-1 * dvec), G=(-1 * Amat.transpose()), h=(-1 * bvec), 
                       lb=np.full((1, m), (-.15)), ub=np.full((1, m), .5), 
                       solver='quadprog')
        sdP[i] = math.sqrt(np.sum(res * means.to_numpy()))
        weights[i, :] = res
    
    print(weights)
    
    
    return res_max
    
    
    
    
getMinVarPort(syms)