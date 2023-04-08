#%%

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
from scipy.optimize import curve_fit

#%%
"""
Settings
"""
sns.set_style('darkgrid')

plt.rcParams["font.family"] = "serif"

bbox = dict(boxstyle ="round",facecolor='white')

__Fpath__="../../"


#%%
"""
Utilities
"""

def fun1(x, m):
    return  m * x

def fun2(x,a,b,e):
    return a*np.power(x,b)+e

#%%

url="Data\Intermedio\CSV_Efecto_Hall\Acrividad_1.csv"

df = pd.read_csv(__Fpath__+url)

# %%

df = df[df["Corriente (A)"] < 1.02]
x = df["Corriente (A)"] 
y = df["Campo magnetico Real (mT)"]

h=(max(x)-min(x))/200

x_grid = np.arange(min(x),max(x),h)

param, param_cov = curve_fit(fun1, x, y)

sterr = np.sqrt(np.diag(param_cov))

print(sterr)

y_comparation = param[0]*x

df["residuales"] = y - y_comparation

df["residuales"] = df["residuales"]/(max(abs(df["residuales"])))

# %%

"""
plotting
"""
fig, axs = plt.subplots(ncols=1,nrows=2,
                        gridspec_kw={'height_ratios':[3,1]},
                        figsize=(8,8),sharex=True)

plt.subplots_adjust(hspace=0.01)


ax1 = axs[0]
ax2 = axs[1]


ax1.errorbar(x="Corriente (A)",y="Campo magnetico Real (mT)",data=df,
             xerr=0.01, yerr=0.1,
             fmt=" ",alpha=0.5,color="#1ECBE1",
             label="Barras de error")

sns.scatterplot(data=df, x="Corriente (A)", y="Campo magnetico Real (mT)",
                color="#1ECBE1", label="Datos experimentales",
                ax=ax1)

ax1.plot(x_grid, param*x_grid,
        label='regresion',linestyle="dashed",
        color="#E1341E"
        )

ax1.annotate("m = {:.0f} +/- {:.0f} mT/A".format(param[0],param_cov[0][0]), 
             xy=(0.0, 100.0),
             bbox = bbox)

ax1.legend()
ax1.set_ylabel("Campo magnetico (mT)")
ax1.set_title("Campo magnetico vs Corriente")


sns.scatterplot(data=df, x="Corriente (A)",y="residuales",
                color="#1ECBE1", label="error normalizado")

ax2.axhline(0,0,5000,ls ="--", 
           color="#E1341E")
ax2.set_ylabel("error normalizado")
ax2.set_xlabel("Corriente (A)")

#set the path file
plt.savefig(__Fpath__+"Images/Actidad1.png",dpi=600)


# %%

"""
Actividad 3
"""
#corriente ip=30mA
url="Data\Intermedio\CSV_Efecto_Hall\Acrividad_3-1.csv"

df = pd.read_csv(__Fpath__+url)

#%%

L = 20E-3
A = 10E-6

df["campo"] = round(df["corriente (mA)"]*param[0],1)

r_0 = round(df["voltaje longitudinal (v)"][0]/30E-3,1) 

sigma_0 = round(L/(r_0*A),1)

df["Resistencia"] = round(df["voltaje longitudinal (v)"]/30E-3,1)

df["Diferencia"] = (df["Resistencia"] - r_0)/r_0
#%%


df = df[df["Diferencia"] > 0.0]
x = df["campo"] 
y = df["Diferencia"]

h=(max(x)-min(x))/200

x_grid = np.arange(min(x),max(x),h)

param, param_cov = curve_fit(fun2, x, y)

sterr = np.round(np.sqrt(np.diag(param_cov)),3)

param = np.round(param,3)
print(sterr)
print(param)

y_comparation = param[0]*np.power(x,param[1])+param[2]

df["residuales"] = y - y_comparation

df["residuales"] = df["residuales"]/(max(abs(df["residuales"])))

# %%
"""
plotting
"""
fig, axs = plt.subplots(ncols=1,nrows=2,
                        gridspec_kw={'height_ratios':[3,1]},
                        figsize=(8,8),sharex=True)

plt.subplots_adjust(hspace=0.01)


ax1 = axs[0]
ax2 = axs[1]


ax1.errorbar(x="campo",y="Diferencia",data=df,
             xerr=0.1, yerr=0.01,
             fmt=" ",alpha=0.5,color="#1ECBE1",
             label="Barras de error")

sns.scatterplot(data=df, x="campo", y="Diferencia",
                color="#1ECBE1", label="Datos experimentales",
                ax=ax1)

ax1.plot(x_grid, param[0]*np.power(x_grid,param[1])+param[2],
        label='regresion',linestyle="dashed",
        color="#E1341E"
        )

ax1.annotate("a = {:.0f} +/- {:.0f} mT/A".format(param[0],param_cov[0][0]), 
             xy=(0.0, 0.0),
             bbox = bbox)

ax1.legend()
ax1.set_ylabel("Cambio de resistencia normalizado")
ax1.set_title("Cambio de resistencia vs Campo")


sns.scatterplot(data=df, x="campo",y="residuales",
                color="#1ECBE1", label="error normalizado")

ax2.axhline(0,0,5000,ls ="--", 
           color="#E1341E")
ax2.set_ylabel("error normalizado")
ax2.set_xlabel("Campo (mT)")

#set the path file
plt.savefig(__Fpath__+"Images/Actidad2.png",dpi=600)


# %%
