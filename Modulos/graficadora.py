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

bbox = dict(boxstyle ="round",facecolor='white')

#%%

def graficadorabarreror(df, xkey, ykey, xlabel, 
                        ylabel, x_grid, title, regresion):
    fig, axs = plt.subplots(ncols=1,nrows=2,
                            gridspec_kw={'height_ratios':[3,1]},
                            figsize=(8,8),sharex=True)

    plt.subplots_adjust(hspace=0.01)


    ax1 = axs[0]
    ax2 = axs[1]


    ax1.errorbar(x=xkey,y=ykey,data=df,
                xerr=0.01, yerr=0.1,
                fmt=" ",alpha=0.5,color="#1ECBE1",
                label="Barras de error")

    sns.scatterplot(data=df, x=xkey, y=ykey,
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
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)

    ax2.axhline(0,0,5000,ls ="--", 
            color="#E1341E")
    ax2.set_ylabel("error normalizado")
    ax2.set_xlabel(xkey)


def lineplot(df, xkey, ykey, xlabel, 
                        ylabel, title, labels):
    f,ax=plt.subplots(figsize=(20,10))

    sns.lineplot(data = df, x = xkey, y = ykey,
                 color = "#1ECBE1", label= labels)
    
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    return f,ax
