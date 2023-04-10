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

plt.rcParams["font.size"] = "14"

bbox = dict(boxstyle ="round",facecolor='white')

__Fpath__="../../"

pendiente = 255.00

#%%
"""
Utilities
"""

def fun1(x, m):
    return  m * x

def fun2(x,a,b,e):
    return a*np.power(x,b)+e

def errorC(x,m,dx,dm):
    factor1 = (m*dm)**2
    factor2 = (x*dx)**2
    return np.sqrt(factor1+factor2)

def error_1_0(x,I,dv,dI):
    factor1 = (1/I*dv)**2
    factor2 = np.power((x/(I**2)*dI),2)
    return np.sqrt(factor1+factor2)

def error_1_1(x,r_0,dx,dr0):
    factor1 = np.power(dx/r_0,2)
    factor2 = np.power(dr0*x/(r_0**2),2)
    return np.sqrt(factor1+factor2)

def error_2_1(x,I,L,A,dx,dI):
    factor1 = np.power(L/(A*x)*dI,2)
    factor2 = np.power(L*I/(A*np.power(x,2))*dx,2)
    return np.sqrt(factor1+factor2)

def error_3_1(x,I,B,dx,dI,dB):
    factor1 = np.power(dx/(I*B),2)
    factor2 = np.power(dI*x/((I**2)*B),2)
    factor3 = np.power(dB*x/(I*(B**2)),2)
    return np.sqrt(factor1 + factor2 + factor3)


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
url="Data\Intermedio\CSV_Efecto_Hall\Acrividad_3-2.csv"

df = pd.read_csv(__Fpath__+url)

#%%

L = 20E-3
A = 10E-6

df["campo"] = round(df["corriente (mA)"]*pendiente,1)
df["sterrc"] = errorC(df["campo"],pendiente,0.1,2)

r_0 = round(df["voltaje longitudinal (v)"][0]/30E-3,1) 
dr0 = error_1_0(r_0,30E-3,0.01,1E-3)

sigma_0 = round(L/(r_0*A),1)



df["Resistencia"] = round(df["voltaje longitudinal (v)"]/30E-3,1)
df["sterrR"] = error_1_0(df["voltaje longitudinal (v)"],30E-3,0.01,1E-3)

df["Diferencia"] = (df["Resistencia"] - r_0)/r_0
df["sterrD"] = error_1_1(df["Diferencia"],r_0,df["sterrR"],dr0)


#%%


df = df[df["Diferencia"] > 0.0]
df = df[df["Diferencia"] < max(df["Diferencia"])]
x = df["campo"] 
y = df["Diferencia"]

h=(max(x)-min(x))/200

x_grid = np.arange(min(x),max(x),h)

param, param_cov = curve_fit(fun2, x, y)

sterr = np.sqrt(np.diag(param_cov))

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
             xerr=df["sterrc"], yerr=df["sterrD"],
             fmt=" ",alpha=0.5,color="#1ECBE1",
             label="Barras de error")

sns.scatterplot(data=df, x="campo", y="Diferencia",
                color="#1ECBE1", label="Datos experimentales",
                ax=ax1)

ax1.plot(x_grid, param[0]*np.power(x_grid,param[1])+param[2],
        label='regresion',linestyle="dashed",
        color="#E1341E"
        )

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

"""
Actividad 4
"""

url="Data\Intermedio\CSV_Efecto_Hall\Actvidad_4-1_TarjetaN.csv"

df = pd.read_csv(__Fpath__+url)

#%%
"""
Placa N
"""

L = 20E-3
A = 10E-6

Ip = 30E-3

df["conductividad"] = L*Ip/(df["Voltaje Longitudinal (V)"]*A)
df["sterrc"] = error_2_1(df["Voltaje Longitudinal (V)"],Ip,L,A,0.01,1E-3)

df["Inverso T"] = 1/(df["Temperatura (Cº)"]+273.15)
df["sterrT"] = 1/np.power(df["Temperatura (Cº)"]+273.15,2)

#%%

df["lnc"] = np.log(df["conductividad"])
df["sterrlnc"] = df["sterrc"]/df["conductividad"]

#%%

df = df[df["Inverso T"]<0.0030]
df = df[df["Inverso T"]>0.0027]
x = df["Inverso T"]
y = df["lnc"]

lin = stats.linregress(x,y)
h=(max(x)-min(x))/200

x_grid = np.arange(min(x),max(x),h)

m = lin.slope

dm = lin.stderr

y_comparation = lin.intercept + lin.slope*x

df["residuales"] = y - y_comparation
df["residuales"] = df["residuales"]/(max(abs(df["residuales"])))

# %%

fig, axs = plt.subplots(ncols=1,nrows=2,
                        gridspec_kw={'height_ratios':[3,1]},
                        figsize=(8,8),sharex=True)

plt.subplots_adjust(hspace=0.01)


ax1 = axs[0]
ax2 = axs[1]


ax1.errorbar(x="Inverso T",y="lnc",data=df,
             xerr=df["sterrT"], yerr=df["sterrlnc"],
             fmt=" ",alpha=0.5,color="#1ECBE1",
             label="Barras de error")

sns.scatterplot(data=df, x="Inverso T", y="lnc",
                color="#1ECBE1", label="Datos experimentales",
                ax=ax1)

ax1.plot(x_grid, m*x_grid+lin.intercept,
        label='regresion',linestyle="dashed",
        color="#E1341E"
        )
ax1.annotate("m = {:.0f} +/- {:.0f} (V*K/(J*m)) ".format(m,dm), xy=(0.0027, 5.67),
             bbox = bbox)

ax1.legend()
ax1.set_ylabel("conductividad (ohm^-1 * m^-1)")
ax1.set_title("Conductividad vs inverso de T en escala semilogaritmica")


sns.scatterplot(data=df, x="Inverso T",y="residuales",
                color="#1ECBE1", label="error normalizado")

ax2.axhline(0,0,5000,ls ="--", 
           color="#E1341E")
ax2.set_ylabel("error normalizado")
ax2.set_xlabel("Inverso T (K^-1)")

#set the path file
plt.savefig(__Fpath__+"Images/Actidad_4-1_N.png",dpi=600)

# %%

"""
Placa I
"""

url="Data\Intermedio\CSV_Efecto_Hall\Actvidad_4-1_TarjetaI.csv"

df = pd.read_csv(__Fpath__+url)

#%%

L = 20E-3
A = 10E-6

Ip = 5E-3

df["conductividad"] = L*Ip/(df["Voltaje Longitudinal (V)"]*A)
df["sterrc"] = error_2_1(df["Voltaje Longitudinal (V)"],Ip,L,A,0.01,1E-3)

df["Inverso T"] = 1/(df["Temperatura (Cº)"]+273.15)
df["sterrT"] = 1/np.power(df["Temperatura (Cº)"]+273.15,2)

#%%

df["lnc"] = np.log(df["conductividad"])
df["sterrlnc"] = df["sterrc"]/df["conductividad"]

#%%

x = df["Inverso T"]
y = df["lnc"]

lin = stats.linregress(x,y)
h=(max(x)-min(x))/200

x_grid = np.arange(min(x),max(x),h)

m = lin.slope

dm = lin.stderr

y_comparation = lin.intercept + lin.slope*x

df["residuales"] = y - y_comparation
df["residuales"] = df["residuales"]/(max(abs(df["residuales"])))

# %%

fig, axs = plt.subplots(ncols=1,nrows=2,
                        gridspec_kw={'height_ratios':[3,1]},
                        figsize=(8,8),sharex=True)

plt.subplots_adjust(hspace=0.01)


ax1 = axs[0]
ax2 = axs[1]


ax1.errorbar(x="Inverso T",y="lnc",data=df,
             xerr=df["sterrT"], yerr=df["sterrlnc"],
             fmt=" ",alpha=0.5,color="#1ECBE1",
             label="Barras de error")

sns.scatterplot(data=df, x="Inverso T", y="lnc",
                color="#1ECBE1", label="Datos experimentales",
                ax=ax1)

ax1.plot(x_grid, m*x_grid+lin.intercept,
        label='regresion',linestyle="dashed",
        color="#E1341E"
        )
ax1.annotate("m = {:.0f} +/- {:.0f} (V*K/(J*m)) ".format(m,dm), xy=(0.0026, 1.5),
             bbox = bbox)

ax1.legend()
ax1.set_ylabel("conductividad (ohm^-1 * m^-1)")
ax1.set_title("Conductividad vs inverso de T en escala semilogaritmica")


sns.scatterplot(data=df, x="Inverso T",y="residuales",
                color="#1ECBE1", label="error normalizado")

ax2.axhline(0,0,5000,ls ="--", 
           color="#E1341E")
ax2.set_ylabel("error normalizado")
ax2.set_xlabel("Inverso T (K^-1)")

#set the path file
plt.savefig(__Fpath__+"Images/Actidad_4-1_I.png",dpi=600)

# %%

""""
Actvidad 4.2
"""

"""
Placa N
"""

url="Data\Intermedio\CSV_Efecto_Hall\Actvidad_4-2_TarjetaN.csv"

df = pd.read_csv(__Fpath__+url)

#%%

L = 20E-3
A = 10E-6

B = 1.5*pendiente
dB = errorC(1.5,pendiente,0.01,2)
Ip = 30E-3

df["Rh"] = df["Voltaje Hall (V)"]/(B*Ip)
df["sterrRh"] = error_3_1(df["Voltaje Hall (V)"],Ip,B,0.001,1E-3,dB)

df["Inverso T"] = 1/(df["Temperatura (Cº)"]+273.15)
df["sterrT"] = 1/np.power(df["Temperatura (Cº)"]+273.15,2)

#%%
df = df[ df["Inverso T"]<0.00295 ]

#%%

df["RhT"] = df["Rh"] * np.power((df["Temperatura (Cº)"]+273.15),3/2) 

#%%

sns.scatterplot(data=df, x="Inverso T", y="Rh",
                color="#1ECBE1", label="Datos experimentales",
                )
# %%
