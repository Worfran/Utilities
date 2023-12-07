#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
import os
from scipy import signal


#%%
"""
Settings
"""
sns.set_style('darkgrid')

plt.rcParams["font.family"] = "serif"

plt.rcParams["font.size"] = "16"

bbox = dict(boxstyle ="round",facecolor='white')

__Fpath__="../../"

folder_path1 =  __Fpath__ + "Data/Proyecto experimental/Datos_Avila/Datos MoS2/50kV/Interes/"

folder_path2 =  __Fpath__ + "Data/Proyecto experimental/Datos_Avila/Datos MoS2/50kV/Interes2/"

#%%

"""
Funciones
"""

def reader(file):
    df = pd.read_csv(file, delimiter=",",  
                    header=None, error_bad_lines=False)
    columname = file_path[-10:]
    df.rename(columns={0: columname}, inplace=True)
    

    return df, columname

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

#%%
"""
Data
"""
folder_paths = [folder_path1, folder_path2]

df = pd.DataFrame()
columnames = []
for folder_path in folder_paths:
    for filename in os.listdir(folder_path):
        file_path = folder_path+filename
        df1, columname=reader(file_path)
        df1 = df1.loc[30:600]
        df[columname] = df1[columname]
        columnames.append(columname)

# %%
print(columnames)
# %%
