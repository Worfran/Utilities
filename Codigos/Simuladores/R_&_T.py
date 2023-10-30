#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


#%%
"""
Settings
"""
sns.set_style('darkgrid')

plt.rcParams["font.family"] = "serif"

plt.rcParams["font.size"] = "12"


__Fpath__="../../"

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

df = pd.DataFrame()

df['T'] = arr_t
df['R'] = arr_r
df['thetaI'] = thetaI

fig = plt.figure()
sns.scatterplot(data=df, x='thetaI',y='T',
                color="#1ECBE1",label='Coeficiente de transmicion')
sns.scatterplot(data=df, x='thetaI',y='R',
                color="#E1341E",label='Coeficiente de reflexion')

plt.xlabel("Angulo Incidente (rad)")
plt.ylabel("Coeficiente")
plt.title("Coeficiente vs angulo de incidencia")
plt.legend()

plt.savefig(__Fpath__+"Images/Electro2_T3.png",dpi=600)





# %%
