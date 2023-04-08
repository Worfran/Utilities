#%%

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

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
Set the file
"""
url ="Data/Intermedio/Muones/toma1.data"

file = __Fpath__+url

#%%
"""
Reading 
"""


df = pd.read_csv(file, delimiter=" ",  
                 header=None, error_bad_lines=False)

#%%

"""
Filtering
"""

#Apartit de cuantos segundos quiere filtrar
t=5000
df = df[df[0] <t]




# %%

"""
Processing
"""

dfp = df[0].value_counts()

dfp = pd.DataFrame({"time":dfp.index.values , "frequency":dfp.values})



sns.scatterplot(data=dfp, x="time", y="frequency",
                color="#1ECBE1", label="Datos experimentales")


#%%

#segunfo filtro
ff=100
dfp = dfp[dfp["frequency"] < ff ]

dfp["logscale"] = np.log(dfp["frequency"]/max(dfp["frequency"])) 

x = dfp["time"]
y = dfp["logscale"]

lin = stats.linregress(x,y)
h=(max(x)-min(x))/200

x_grid = np.arange(min(x),max(x),h)

m = round(lin.slope*1E3,2)

dm = round(lin.stderr*1E3,2)

y_comparation = lin.intercept + lin.slope*x

dfp["residuales"] = y - y_comparation

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


ax1.errorbar(x="time",y="logscale",data=dfp,
             xerr=0.1,
             fmt=" ",alpha=0.5,color="#1ECBE1",
             label="Barras de error")

sns.scatterplot(data=dfp, x="time", y="logscale",
                color="#1ECBE1", label="Datos experimentales",
                ax=ax1)

ax1.plot(x_grid, lin.intercept + lin.slope*x_grid,
        label='regresion',linestyle="dashed",
        color="#E1341E"
        )

ax1.annotate("m = ({} +/- {}) (μs)^-1 ".format(m,dm), xy=(0.0, -1.0),
             bbox = bbox)

ax1.legend()
ax1.set_xlabel("Tiempo (μs)")
ax1.set_ylabel("Frecuencia de ocurrencia del dato")
ax1.set_title("Frecuencia normalizada de los tiempos en escala semilogaritmica (1era toma)")


sns.scatterplot(data=dfp, x="time",y="residuales",
                color="#1ECBE1", label="error")

ax2.axhline(0,0,5000,ls ="--", 
           color="#E1341E")
ax2.set_ylabel("error")
ax2.set_xlabel("Tiempo (μs)")

#set the path file
plt.savefig(__Fpath__+"Images/toma1.png",dpi=600)


# %%
