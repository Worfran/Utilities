#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
sns.set_style('darkgrid')
#%%
__Ipath__= "C:/Users/frawo\Documents/Programacion/Utilities/Images/"

"""
Funciones Auxiliares
"""

def Bragg(theta,d,n):
    factor1 = 2*d* np.sin(theta)
    return factor1 / n

def errorS1(d,x:any):
    factor1 = np.sqrt((2*d* np.cos(x)*0.1)**2)
    return factor1

def normlog(y,y_0):
    return np.log(y/y_0)

#%%
"""
Actividad 1
"""

url="C:/Users/frawo/Documents/Programacion/Utilities/Data/Intermedio/CSV/Datos_1.csv"

df = pd.read_csv(url,
    names=["Angulo", "Intensidad"],
    dtype={"Angulo":float, "Intensidad":float},
    skiprows=[0,1,2]
    )

theta = df["Angulo"]


lam = 1E9* Bragg(np.radians(theta), 2.014E-10, 1)

df["Longitud de onda"] = lam

err = errorS1(2.014E-10, np.radians(theta))

derror = pd.DataFrame()

derror["ERROR"] =  10E9*err


#%%

"""
Graficacion 
"""
#eliminar estas comillas para generar graficas
"""
_, axes = plt.subplots(1,2, figsize = (15, 5))

axs1 = axes[0]
axs2 = axes[1]

sns.scatterplot(
    data=df, x="Angulo", y="Intensidad",
    ax=axs1, s=8, color="#1ECBE1")
axs1.set_title('Intensidad vs Angulo ')
axs1.set_xlabel('Ángulo de insidencia +/1 (°) ')
axs1.set_ylabel('Intensidad +/- 1 (Imp/s)')
axs1.errorbar(df["Longitud de onda"], df["Intensidad"] , xerr=0.1, fmt=' ' )

sns.scatterplot(
    data=df, x="Longitud de onda", y="Intensidad",
    ax=axs2, s=8, color="#1ECBE1")
axs2.set_title('Intensidad vs Longitud de onda')
axs2.set_xlabel('Longitud de onda +/- 0.1 (nm)')
axs2.set_ylabel('Intensidad +/- 1 (Imp/s)')
axs2.errorbar(df["Longitud de onda"], df["Intensidad"] , yerr=derror["ERROR"], fmt=' ' )


_.savefig(__Ipath__+'Actividad 1.png',dpi=1200)

"""
#eliminar estas comillas para generar graficas
# %%
"""
Actividad 2
"""

url1="C:/Users/frawo/Documents/Programacion/Utilities/Data/Intermedio/CSV/Datos_2.csv"

df1 = pd.read_csv(url1,
        names= ["Angulo","0.02 mm","0.04 mm","0.06 mm","0.08 mm","0.1 mm","control"],
        dtype={"Angulo":float, "0.02 mm": float,"0.04 mm":float,
               "0.06 mm":float,"0.08 mm":float,"0.1 mm":float, "control":float},
        skiprows=[0,1,2])

#%%

"""
Graficas A21
"""
#eliminar estas comillas para generar graficas
"""
plt.figure()

sns.scatterplot(
    data=df1, y="0.02 mm", x="Angulo",
    label="0.02 mm")

sns.scatterplot(
    data=df1, y="0.04 mm", x="Angulo",
    label="0.04 mm")

sns.scatterplot(
    data=df1, y="0.06 mm", x="Angulo",
    label="0.06 mm")

sns.scatterplot(
    data=df1, y="0.08 mm", x="Angulo",
    label="0.08 mm")

sns.scatterplot(
    data=df1, y="0.1 mm", x="Angulo",
    label="0.1 mm")

sns.scatterplot(
    data=df1, y="control", x="Angulo",
    label="control")


plt.legend()
plt.ylabel("Intensidad +/- 1 (Imp/s)")
plt.xlabel("Ángulo de incidencia +/- 0.1 (°)")
plt.title("Intensidad vs Angulo de incidencia (variando espesor de muestra Al)")
plt.savefig(__Ipath__ + "Actividad 2-0.png", dpi=1200)
"""
#eliminar estas comillas para generar graficas
#%%

"""
Procesando datos
"""

theta = df1["Angulo"]

lam = 1E9* Bragg(np.radians(theta), 2.014E-10, 1)


df1["Longitud de onda"] = lam

E1 = np.array(10*[0.02])
E2 = np.array(10*[0.04])
E3 = np.array(10*[0.06])
E4 = np.array(10*[0.08])
E5 = np.array(10*[0.1])
E6 = np.array(10*[0])


lam = np.array(lam)

I_002 = np.log(np.array(df1["0.02 mm"]/df1["0.02 mm"].iloc[0]))
I_004 = np.log(np.array(df1["0.04 mm"]/df1["0.04 mm"].iloc[0]))
I_006 = np.log(np.array(df1["0.06 mm"]/df1["0.06 mm"].iloc[0]))
I_008 = np.log(np.array(df1["0.08 mm"]/df1["0.08 mm"].iloc[0]))
I_01 = np.log(np.array(df1["0.1 mm"]/df1["0.1 mm"].iloc[0]))
I_control = np.log(np.array(df1["control"]/df1["control"].iloc[0]))

I = np.round(np.concatenate((I_control,I_002,I_004,I_006,I_008,I_01),axis=0 ),3)

lamG = np.round(np.concatenate((lam,lam,lam,lam,lam,lam), axis=0),3)

Eg = np.concatenate((E6,E1,E2,E3,E4,E5), axis=0)


d = {"Intensidad": I, "Longitud de onda": lamG, "Espesor" : Eg}


df11= pd.DataFrame(d)

#%%

"""
Graficar A22
"""
#eliminar estas comillas para generar graficas
"""
plt.figure()
sns.scatterplot(data=df11, 
    x="Espesor", y="Intensidad", 
    hue="Longitud de onda", palette="deep")

plt.xlabel("grosor en mm")
plt.ylabel("Intensidad +/- 1 (Imp/s)")
plt.title("Intensidad de cada longitud de onda para distintos espesores")
plt.savefig(__Ipath__ + "Actividad 2-2.png",dpi=1200)
"""
#eliminar estas comillas para generar graficas

# %%
"""
Regresiones lineales
"""
lam = df11["Longitud de onda"].iloc[0:9]
x0 = np.array(df11["Espesor"][df11["Longitud de onda"] == lam[0]]).reshape(-1,1)

Mus = np.zeros(len(lam))

for i in range(len(lam)):
    yi = np.array(df11["Intensidad"][df11["Longitud de onda"] == lam[i]])
    model_i = LinearRegression().fit(x0,yi)
    Mus[i] = model_i.coef_



x = 1E3*np.power(np.array(lam),3)


d = {"Espesor": x, "Mus": Mus}

df12 = pd.DataFrame(d)

poly = PolynomialFeatures(degree=3)

x_poly = poly.fit_transform(x.reshape(-1,1))

poly.fit(x_poly, Mus)

lin2 = LinearRegression().fit(x_poly,Mus)

x_grid0 = np.arange(min(x),max(x),5.3E-3)
x_grid = x_grid0.reshape(len(x_grid0),1)
y_grid = lin2.predict(poly.fit_transform(x_grid))

df13 = {"x regression" : x_grid0, "y regression": y_grid}

df13 = pd.DataFrame(df13)

#%%

"""
Graficar A3
"""
#eliminar estas comillas para generar graficas
"""
plt.figure()
sns.scatterplot(data=df12, 
    x="Espesor", y="Mus", label="Valores experimentales", color="#1ECBE1")

sns.lineplot(data=df13, 
    x="x regression", y="y regression", label="Valores regresion (Grado 3)"
    ,color="#E1341E", linestyle="dashed" )

plt.xlabel(" λ [pm] ")
plt.ylabel(" μ ")
plt.title(" Coenficientes de absorcion vs Longitude de onda  ")
plt.savefig(__Ipath__ + "Actividad 2-3.png", dpi=1200)

"""
#eliminar estas comillas para generar graficas

# %%

"""
Actividad 3
"""


