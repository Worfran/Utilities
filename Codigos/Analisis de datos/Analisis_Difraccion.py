#%%
"""
Imports
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

#%%

"""
Setings
"""
sns.set_style('darkgrid')

bbox = dict(boxstyle ="round",facecolor='white')

__Fpath__="../../"

#%%

"""
Funciones
"""

def lineal(x,m):
    return m*x

def sterriU(x,dx):
    factor1 = 1/(np.power(np.sqrt(x),3))
    return abs(1/2*factor1) 

#%%

file = "Data/Intermedio/CSV_Difraccion/CSV_datos.csv"

df = pd.read_csv(__Fpath__+file)

# %%


x = 1/(np.sqrt(df['U(k)'])) 
df['Inversa de U'] = x
D1 = df['D1(cm)']*1E-2
D2 = df['D2(cm)']*1E-2

df['D1 (m)'] = D1
df['D2 (m)'] = D2 

h=(max(x)-min(x))/200

x_grid = np.arange(min(x),max(x),h)

param1, param_cov1 = curve_fit(lineal, x, D1)
param2, param_cov2 = curve_fit(lineal, x, D2)

sterr1 = np.sqrt(np.diag(param_cov1))
sterr2 = np.sqrt(np.diag(param_cov2))


y_comparation1 = param1[0]*x
y_comparation2 = param2[0]*x

df["residuales1"] = D1 - y_comparation1
df["residuales2"] = D2 - y_comparation2

df["residuales1"] = df["residuales1"]/(max(abs(df["residuales1"])))
df["residuales2"] = df["residuales2"]/(max(abs(df["residuales2"])))
df["staderrv"] = sterriU(df['U(k)'],0.01)
#%%
"""
plotting
"""
fig, axs = plt.subplots(ncols=1,nrows=2,
                        gridspec_kw={'height_ratios':[3,1]},
                        figsize=(8,8),sharex=True)

plt.subplots_adjust(hspace=0.01)


ax1 = axs[0]
ax2 = axs[1]


ax1.errorbar(x='Inversa de U',y='D2 (m)',data=df,
             xerr=df['staderrv'], yerr=0.001,
             fmt=" ",alpha=0.5,color="#1ECBE1",
             label="Barras de error")

sns.scatterplot(data=df, x='Inversa de U', y='D2 (m)',
                color="#1ECBE1", label="Datos experimentales",
                ax=ax1)

ax1.plot(x_grid, param2[0]*x_grid,
        label='regresion',linestyle="dashed",
        color="#E1341E"
        )

ax1.text(0.45,0.0500,'m = {:.2e} +/- {:.0e}'.format(param2[0],sterr2[0]))
ax1.legend()
ax1.set_ylabel("Radio D2 (m)")
ax1.set_title("Radio 2 vs U^(-1/2)")


sns.scatterplot(data=df, x='Inversa de U',y='residuales2',
                color="#1ECBE1", label="error normalizado")

ax2.axhline(0,0,5000,ls ="--", 
           color="#E1341E")
ax2.set_ylabel("error normalizado")
ax2.set_xlabel("U^(-1/2) (V^-1)")

#set the path file
plt.savefig(__Fpath__+"Images/Difraccion_A3D_2.png",dpi=600)

# %%
