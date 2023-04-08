#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

sns.set_style('darkgrid')

bbox = dict(boxstyle ="round",facecolor='white')

__Fpath__="../../"

#%%

"""
Actividad 1
"""

url= "Data/Intermedio/CSV_Torsion/Actividad_.csv"


#%%

"""
Actividad 3
"""
url=__Fpath__+"Data/Intermedio/CSV_Torsion/Actividad_3.csv"

df3 = pd.read_csv(url, header=0)

angulos = df3["angulo "]

dangulos = np.absolute(angulos-3.00)

df3["delta angulo"] = dangulos

# %%

"""
Graficar 3-1
"""

plt.figure()

sns.scatterplot(data=df3, x="corriente ", y="delta angulo",
                color="#1ECBE1", label="Datos experimentales"
                )

plt.xlabel("Corriente +/- 0.01 (A)")
plt.ylabel("Delta de ángulo +/- 0.01 (rad)")
plt.title("Delta de ángulo vs Corriente")

plt.savefig(__Fpath__+"Images/Actividad_3-1.png",dpi=600)

# %%

dregresion = df3[df3["delta angulo"] < 0.605]

x = dregresion["corriente "]
y = dregresion["delta angulo"]

lin3 = stats.linregress(x, y)

h=(max(x)-min(x))/200

x_grid = np.arange(min(x),max(x),h)

m = round(lin3.slope,2)

dm = round(lin3.stderr,2)

"""
residuales
"""

y_comparation = lin3.intercept + lin3.slope*x

dregresion["residuales"] = y - y_comparation

"""
Graficar 3-2
"""

fig, axs = plt.subplots(ncols=1,nrows=2,
                        gridspec_kw={'height_ratios':[3,1]},
                        figsize=(8,8),sharex=True)

plt.subplots_adjust(hspace=0.01)


ax1 = axs[0]
ax2 = axs[1]


ax1.errorbar(x="corriente ",y="delta angulo",data=dregresion,
             yerr=0.1,xerr=0.01,
             fmt=" ",alpha=0.5,color="#1ECBE1",
             label="Barras de error")

sns.scatterplot(data=dregresion, x="corriente ", y="delta angulo",
                color="#1ECBE1", label="Datos experimentales",
                ax=ax1)

ax1.plot(x_grid, lin3.intercept + lin3.slope*x_grid,
        label='regresion',linestyle="dashed",
        color="#E1341E"
        )

ax1.annotate("m = ({} +/- {}) rad/A".format(m,dm), xy=(0.0, 0.4),
             bbox = bbox)

ax1.legend()
ax1.set_ylabel("Delta de ángulo (rad)")
ax1.set_title("Delta de ángulo vs Corriente")


sns.scatterplot(data=dregresion, x="corriente ",y="residuales",
                color="#1ECBE1", label="error")

ax2.axhline(0,0,1.05,ls ="--", 
           color="#E1341E")
ax2.set_ylabel("residuales")
ax2.set_xlabel("Corriente (A)")

plt.savefig(__Fpath__+"Images/Actividad_3-2.png",dpi=600)

# %%

"""
Actividad Extra
"""

url = __Fpath__ + "Data/Intermedio/CSV_Torsion/Actividad_ext-1.csv"

df41 = pd.read_csv(url, header=0)

maxi1 = max(df41["Amplitud mV"])


w1 = df41[df41['Amplitud mV'] == maxi1]

w1 = w1["frecuencia Hz"].values


#%%
"""
Graficar E1
"""


plt.figure()


plt.errorbar(x="frecuencia Hz",y="Amplitud mV",data=df41,
             yerr=1E-6,xerr=1E-9,
             fmt=" ",alpha=1,color="#1ECBE1",
             label="Barras de error")

sns.scatterplot(data=df41,x="frecuencia Hz", y="Amplitud mV",
                color="#1ECBE1", label="Datos experimentales"
                )

plt.vlines(w1, 0, 5000, linestyles ="dotted", 
           color="#E1341E", label="w1"
           )

plt.annotate("w1 = {}Hz".format(w1[0]), xy=(0.5, 3500),
             bbox = bbox)

plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud (mV)")
plt.title("Amplitud vs Frecuencia primera configuración")
plt.legend()

plt.savefig(__Fpath__+"Images/Actividad_E1.png",dpi=600)

# %%

url = __Fpath__ + "Data/Intermedio/CSV_Torsion/Actividad_ext-2.csv"

df42 = pd.read_csv(url, header=0)


maxi2 = max(df42["Amplitud mV"])


w2 = df41[df42['Amplitud mV'] == maxi2]

w2 = w2["frecuencia Hz"].values



#%%
"""
Graficar E2
"""


fig = plt.figure()

plt.errorbar(x="frecuencia Hz",y="Amplitud mV",data=df42,
             yerr=1E-6,xerr=1E-9,
             fmt=" ",alpha=1,color="#1ECBE1",
             label="Barras de error")

sns.scatterplot(data=df42,x="frecuencia Hz", y="Amplitud mV",
                color="#1ECBE1", label="Datos experimentales"
                )

plt.vlines(w2, 600, 1200, linestyles ="dotted", 
           color="#E1341E", label="w2"
           )

plt.annotate("w2 = {}Hz".format(w2[0]), xy=(0.5, 900),
             bbox = bbox)

plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud (mV)")
plt.title("Amplitud vs Frecuencia segunda configuración")
plt.legend()

plt.savefig(__Fpath__+"Images/Actividad_E2.png",dpi=600)


# %%
