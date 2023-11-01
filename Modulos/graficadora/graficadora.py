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


def plot_dataframe_multiplesubplots(df, x_key, y_keys, x_label, y_labels, plot_labels, axes, title):
    """
    Plot multiple columns from a DataFrame on different subplots using Seaborn.

    Parameters:
    - df: DataFrame containing the data
    - x_key: Key for the x-axis
    - y_keys: List of keys for the y-axes
    - x_label: Label for the x-axis
    - y_labels: List of labels for the y-axes
    - plot_labels: List of labels for each plot in the subplot
    - axes: List of Axes to plot on
    - title: Title for the subplots
    """

    if len(y_keys) != len(axes):
        raise ValueError("Number of y_keys must match the number of axes")

    for i, (y_key, ax) in enumerate(zip(y_keys, axes)):
        sns.scatterplot(x=df[x_key], y=df[y_key], label=plot_labels[i], ax=ax)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_labels[i])
        ax.set_title(f"{title} - {plot_labels[i]}")
        ax.legend()

    plt.tight_layout()

def plot_dataframe(df, x_key, y_keys, x_label, y_label, plot_labels, ax, title):
    """
    Plot multiple columns from a DataFrame on the same subplot using Seaborn.

    Parameters:
    - df: DataFrame containing the data
    - x_key: Key for the x-axis
    - y_keys: List of keys for the y-axes
    - x_label: Label for the x-axis
    - y_labels: List of labels for the y-axes
    - plot_labels: List of labels for each plot in the subplot
    - ax: Axes to plot on
    """

    for i, y_key in enumerate(y_keys):
        sns.lineplot(x=df[x_key], y=df[y_key], label=plot_labels[i], ax=ax)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
