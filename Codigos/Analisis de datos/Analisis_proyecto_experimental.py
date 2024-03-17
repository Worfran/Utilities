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

folder_path =  __Fpath__ + "Data/Proyecto experimental/Datos_Avila/Datos MoS2/50kV/Interes2/"

images_folder_path = __Fpath__ + "Images/Proyecto Experimental/Datos_MoS2/"

file_ch = __Fpath__ + "Data/Proyecto experimental/Datos_Avila/Datos MoS2/Energy_ch.csv"


#%%

"""
Funciones
"""

def reader(file):
    df = pd.read_csv(file, delimiter=",",  
                    header=None,)
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

df = pd.DataFrame()
columnames = []
for filename in os.listdir(folder_path):
    file_path = folder_path+filename
    df1, columname=reader(file_path)
    df1 = df1.loc[30:600]
    df[columname] = df1[columname]
    columnames.append(columname)

#%%
"""
Energy chanel 
"""

df1, columname = reader(file_ch)

x_key = 'ch_energy'
df['ch_energy'] = df1[columname]

print(df)

#%%
"""
Selecting the data
"""
 

print(columnames)

y_keys = [columnames[-2]] + [columnames[-1]] + columnames[:-2]
#y_keys = columnames

print(y_keys)
#%%

"""
Noise reduction 
"""
b, a = signal.butter(3, 0.05)

for key in y_keys:
    y = df[key]
    yy = signal.filtfilt(b, a, y)
    df[key] = yy

#%%
"""
Ploting
"""
n = len(df[columnames[0]])
x = np.arange(0,n,1)

df["canales"] = x

# Create subplots
fig, ax = plt.subplots(figsize=(8, 6))

# Specify keys and labels
#x_key = "canales"
x_label = "Energia [keV]"
y_label = "conteos"
plot_labels = ["filtro", '950nm', '1350nm', '1700nm', '2000nm', '2250nm']
title = "Frecuencia por canal a 50kV"

# Call the function to plot data on multiple axes
plot_dataframe(df, x_key, y_keys, x_label, y_label, plot_labels, ax, title)

"""
Saving image
""" 
name = "placas_Sumadas_MoS2_50kV.png"
#plt.savefig(images_folder_path +  name,dpi=600)


#%%

"""
Atenuation 
"""

