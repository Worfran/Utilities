#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

#%%

#Funciones para la ecuacion

def thetat(thetaI, n1,n2):
    f = n1/n2
    s = np.sin(thetaI)
    return np.arcsin(f*s)

def beta(n1,n2):
    return n1/n2

def alfa(T,I):
    a = np.cos(T)
    b = np.cos(I)
    return a/b

def param(thetaI, n1, n2):
    T = thetat(thetaI, n1,n2)
    af = alfa(T, thetaI) 
    bt =beta(n1,n2)

    return (af,bt)

#%%

#Factores 

def r(thetaI, n1, n2):
    af, bt = param(thetaI, n1, n2)
    f1 = af - bt
    f2 = af + bt 
    return np.power((f1/f2),2)

def t(thetaI, n1, n2):
    af, bt = param(thetaI, n1,n2)
    f1 = af*bt
    f2 = af + bt
    f3 = np.power(2/f2,2)

    return f1*f3

#%%

pi = np.pi
thetaI = np.linspace(0,pi/2)
n1 = 1
n2 =1.5

arr_r = r(thetaI,n1, n2)
arr_t = t(thetaI, n1, n2) 

plt.plot(thetaI, arr_r)
plt.plot(thetaI, arr_t)






# %%
