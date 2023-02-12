#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

sns.set_style('darkgrid')
#%%
__Ipath__= "C:/Users/frawo\Documents/Programacion/Utilities/Images/"

"""
Funciones Auxiliares
"""

eV= 1000*1.602E-19

def maximos(x):
    return None 

def Bragg(theta,d,n):
    factor1 = 2*d* np.sin(theta)
    return factor1 / n

def errorS1(d,x:any):
    factor1 = np.sqrt((2*d* np.cos(x)*0.1)**2)
    return factor1

def normlog(y,y_0):
    return np.log(y/y_0)

def fun(x,a,b,c,d):
    factor = a*np.power(x,3)+b*np.power(x,2)+c*np.power(x,1)+d*np.power(x,0)
    return factor

def fun2(x,a):
    factor1 =a*1*np.sqrt(np.power((x-9.979),3))
    return factor1

def fun3(x,a):
    factor1 =a*1*np.sqrt(np.power(x,3))
    return factor1

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

plt.figure()
sns.scatterplot(data=df11, 
    x="Espesor", y="Intensidad", 
    hue="Longitud de onda", palette="deep")

plt.xlabel("grosor en mm")
plt.ylabel("Intensidad +/- 1 (Imp/s)")
plt.title("Intensidad de cada longitud de onda para distintos espesores")
plt.savefig(__Ipath__ + "Actividad 2-2.png",dpi=1200)


# %%
"""
Regresiones lineales
"""
lam = df11["Longitud de onda"].iloc[0:9]
x0 = np.array(df11["Espesor"][df11["Longitud de onda"] == lam[0]])

Mus = np.zeros(len(lam))
Mus_error = np.zeros(len(lam))

for i in range(len(lam)):
    yi = np.array(df11["Intensidad"][df11["Longitud de onda"] == lam[i]])
    model_i = stats.linregress(x0,yi)
    Mus[i] = round(model_i.slope,2)
    Mus_error[i] = round(model_i.stderr,2)

p = 2.7E3
x = 1E3 * np.power(np.array(lam),3)

x_grid = np.arange(min(x),max(x),4.19E-3)

Mus = Mus / p
Mus_error = Mus_error / p

d = {"Longitud de onda": x, "Mus": Mus}

df12 = pd.DataFrame(d)


popt, pcov = curve_fit(fun, x, Mus)

sterr = np.sqrt(np.diag(pcov))

a,b,c,d = round(popt[0],4),round(popt[1],4),round(popt[2],4),round(popt[3],4)

a_e,b_e,c_e,d_e = round(sterr[0],4),round(sterr[1],4),round(sterr[2],4),round(sterr[3],4),

bbox = dict(boxstyle ="round",facecolor='white')

#%%

"""
Graficar A3
"""
#eliminar estas comillas para generar graficas

plt.figure()
sns.scatterplot(data=df12, 
    x="Longitud de onda", y="Mus", 
    label="Valores experimentales", color="#1ECBE1")

plt.errorbar(df12["Longitud de onda"], df12["Mus"] , yerr=Mus_error, fmt=' ' )

plt.plot( x_grid , fun(x_grid, *popt), 
    label="Valores regresion (Grado 3)"
    ,color="#E1341E", linestyle="dashed")
plt.annotate("a = {} +/- {}; b= {} +/- {}".format(a,a_e,b,b_e), 
             xy=(0.2, 0), bbox = bbox)
plt.annotate("c = {} +/- {}; d= {} +/- {}".format(c,c_e,d,d_e,), 
             xy=(0.2, -0.0002), bbox = bbox)

plt.xlabel(" λ^3 [pm] ")
plt.ylabel(" μ ")
plt.title(" Coenficientes de absorcion vs Longitude de onda  ")
plt.legend()
plt.savefig(__Ipath__ + "Actividad 2-3.png", dpi=1200)

#eliminar estas comillas para generar graficas

# %%

"""
Actividad 3
"""
url21="C:/Users/frawo/Documents/Programacion/Utilities/Data/Intermedio/CSV/Datos_3-1.csv"

df21_v = pd.read_csv(url21,
        names= ["Angulo","11kV","13kV","15kV","17kV","19kV","21kV"
                ,"23kV","25kV","27kV","29kV","31kV","33kV","35kV"],
        dtype={"Angulo":float,"11kV":float,"13kV":float,"15kV":float,"17kV":float
               ,"19kV":float,"21kV":float,"23kV":float,"25kV":float,"27kV":float
               ,"29kV":float,"31kV":float,"33kV":float,"35kV":float},
        skiprows=[0,1,2]) 

voltajes = ["11kV","13kV","15kV","17kV","19kV","21kV"
        ,"23kV","25kV","27kV","29kV","31kV","33kV","35kV"]

i_s = [0,3,6,9,12]

theta = df21_v["Angulo"]

lam = 1E9* Bragg(np.radians(theta), 2.014E-10, 1)

df21_v["Longitud de onda"] = lam

plt.figure()
for i in i_s:
    sns.scatterplot(data= df21_v,
        x = "Longitud de onda", y=voltajes[i], 
        label="Voltaje de {}".format(voltajes[i]), s=8)

plt.xlabel("λ [nm]")
plt.ylabel("Intensidad")
plt.title("Intensidad vs Longitud de onda (voltaje variable)")

#%%
K = df21_v[df21_v["Angulo"] == 20.3]

Is = np.zeros(len(voltajes))
x = np.zeros(len(voltajes))

for i in range(len(voltajes)):
    x[i] = float(voltajes[i].strip("kV"))
    Is[i] = K[voltajes[i]]

h = (max(x) - min(x))/200

x_grid = np.arange(min(x),max(x), h)

popt, pcov = curve_fit(fun2, x, Is) 

sterr = np.sqrt(np.diag(pcov))

d = {"corrientes": x, "Intensidad":Is}

df22_i = pd.DataFrame(d)
bbox = dict(boxstyle ="round",facecolor='white')

sns.scatterplot(data=df22_i,
        x="corrientes", y="Intensidad",
        label="Datos experimentales",color="#1ECBE1")

plt.plot( x_grid , fun2(x_grid, *popt), 
    label="Valores regresion (U_a)"
    ,color="#E1341E", linestyle="dashed")

plt.xlabel("Corriente +/- 0.01 (mA)")
plt.ylabel("Intensidad +/- 1 (Imp/s)")

plt.annotate("b = {} +/- {}".format(round(popt[0],2),round(sterr[0],1)), xy=(15, 8000),
             bbox = bbox)

plt.legend()
plt.title("Intensidad vs Corriente")
plt.savefig( __Ipath__ + "Actividad 3-1.png", dpi=1200)

#%%

popt, pcov = curve_fit(fun3, x, Is) 

sterr = np.sqrt(np.diag(pcov))

d = {"corrientes": x, "Intensidad":Is}

df22_i = pd.DataFrame(d)
bbox = dict(boxstyle ="round",facecolor='white')

sns.scatterplot(data=df22_i,
        x="corrientes", y="Intensidad",
        label="Datos experimentales",color="#1ECBE1")

plt.plot( x_grid , fun2(x_grid, *popt), 
    label="Valores regresion (U_a-U_k varibale)"
    ,color="#E1341E", linestyle="dashed")

plt.xlabel("Corriente +/- 0.01 (mA)")
plt.ylabel("Intensidad +/- 1 (Imp/s)")

plt.annotate("b = {} +/- {}".format(round(popt[0],2),round(sterr[0],1)), xy=(15, 6000),
             bbox = bbox)

plt.legend()
plt.title("Intensidad vs Corriente")
plt.savefig( __Ipath__ + "Actividad 3-2.png", dpi=1200)

# %%
url22 = "C:/Users/frawo/Documents/Programacion/Utilities/Data/Intermedio/CSV/Datos_3-2.csv"

df21_i = pd.read_csv(url22,
        names= ["Angulo","0.1mA","0.2mA","0.3mA","0.4mA","0.5mA","0.6mA"
                ,"0.7mA","0.8mA","0.9mA","1.0mA"],
        dtype={"Angulo":float,"0.1mA":float,"0.2mA":float,"0.3mA":float
               ,"0.4mA":float,"0.5mA":float,"0.6mA":float,"0.7mA":float
               ,"0.8mA":float,"0.9mA":float,"1.0mA":float},
               )

corrientes = ["0.1mA","0.2mA","0.3mA","0.4mA","0.5mA","0.6mA"
        ,"0.7mA","0.8mA","0.9mA","1.0mA"]


i_s=[0,3,6,9]
theta = df21_i["Angulo"]

lam = 1E9* Bragg(np.radians(theta), 2.014E-10, 1)

df21_i["Longitud de onda"] = lam

plt.figure()
for i in i_s:
    sns.scatterplot(
        data=df21_i, 
        x="Longitud de onda", y=corrientes[i],
        label="Corriente de {}".format(corrientes[i]), s=8
    )

plt.xlabel("λ [nm]")
plt.ylabel("Intensidad")
plt.title("Intensidad vs Longitud de onda (corriente variable)")

#%%

K = df21_i[df21_i["Angulo"] == 20.4]

Is = np.zeros(len(corrientes))
x = np.zeros(len(corrientes))

for i in range(len(corrientes)):
    x[i] = float(corrientes[i].strip("mA"))
    Is[i] = K[corrientes[i]]


lin3 = stats.linregress(x,Is)

b = round(1E-2*lin3.slope,2)
b_sterr = round(1E-2*lin3.stderr,2)

d = {"corrientes": x, "Intensidad":Is}

df22_i = pd.DataFrame(d)
bbox = dict(boxstyle ="round",facecolor='white')

sns.scatterplot(data=df22_i,
        x="corrientes", y="Intensidad",
        label="Datos experimentales",color="#1ECBE1")

plt.plot(x, lin3.intercept + lin3.slope*x, 
         label='regresion',linestyle="dashed",
         color="#E1341E")

plt.xlabel("Corriente +/- 0.01 (mA)")
plt.ylabel("Intensidad +/- 1 (Imp/s)")

plt.annotate("b = {} +/- {}".format(b,b_sterr), xy=(0.2, 7000),
             bbox = bbox)

plt.legend()
plt.title("Intensidad vs Corriente")
plt.savefig( __Ipath__ + "Actividad 3-3.png", dpi=1200)


# %%

"""
Actividad 4
"""

url3 = "C:/Users/frawo/Documents/Programacion/Utilities/Data/Intermedio/CSV/Datos_4.csv"

df31 = pd.read_csv(url3,
        names = ["Angulo","13kV","15kV","17kV","19kV"
                ,"21kV","23kV","25kV","27kV","29kV"],
        dtype = {"Angulo":float,"13kV":float,"15kV":float,"17kV":float,"19kV":float
                ,"21kV":float,"23kV":float,"25kV":float,"27kV":float,"29kV":float},
        skiprows=[0,1,2]
                   )

voltajes = ["13kV","15kV","17kV","19kV","21kV"
        ,"23kV","25kV","27kV","29kV"]

i_s = [0,3,6,8]

theta = df21_i["Angulo"]

lam = 1E9* Bragg(np.radians(theta), 2.014E-10, 1)

df31["Longitud de onda"] = lam

#%%
plt.figure()
for i in i_s:
    sns.scatterplot(
        data=df31, 
        x="Longitud de onda", y=voltajes[i],
        label="Voltaje de {}".format(voltajes[i]), s=8
    )
plt.xlabel("λ [nm]")
plt.ylabel("Intensidad")
plt.title("Intensidad vs Longitud de onda (Voltaje variable))")
# %%
