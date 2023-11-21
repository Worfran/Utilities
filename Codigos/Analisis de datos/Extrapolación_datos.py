#%%

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
import os
from scipy import signal
import csv



#%%
"""
Settings
"""
sns.set_style('darkgrid')

plt.rcParams["font.family"] = "serif"

plt.rcParams["font.size"] = "16"

bbox = dict(boxstyle ="round",facecolor='white')

__Fpath__="../../"

folder_path =  __Fpath__ + "Data\Proyecto experimental\Datos_Avila\Datos Grafeno\Calibracion\\"

file_r = __Fpath__ + "Data\Proyecto experimental\Datos_Avila\Datos Grafeno\Calibracion\GEMHV4200-0pF-FG10-CG20-bothOutputswOneUni-3Fe55onDetector-120sLiveTime.csv"

file_w = __Fpath__ + "Data\Proyecto experimental\Datos_Avila\Datos Grafeno\Energy_ch.csv"


"""
Pico más grande 5.9 KeV & segundo 2.9 KeV
"""
#%%

"""
Funciones
"""

def reader(file):
    df = pd.read_csv(file, delimiter=",",  
                    header=None, error_bad_lines=False)
    columname = file[90:96]
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
    df1 = df1.loc[100:5000]
    df[columname] = df1[columname]
    columnames.append(columname)

#%%
columnames = [columnames[0],columnames[3],columnames[5],columnames[1],columnames[4],columnames[6],columnames[2]]
print(columnames)
print(len(columnames))
x = np.arange(0,len(df[columnames[0]]),1)
plt.plot(x,df[columnames[6]])
print(columnames[6])

#2-6
#%%
"""
Noise reduction 
"""
dfE = pd.DataFrame()
columnamesE = []
for i in range(2,7):
    columname = columnames[i]
    columnamesE.append(columname)
    b, a = signal.butter(3, 0.05)
    y = df[columname]
    yy = signal.filtfilt(b, a, y)
    dfE[columname] = yy

print(len(columnamesE))

plt.plot(x,dfE[columnamesE[4]])

#%%

"""
Finding peaks 
"""
index = []
E = [2.9, 5.9]
print(columnamesE)
for i in range(len(columnamesE)):
    columname = columnamesE[i]
    peaks, _ = signal.find_peaks(dfE[columname], height=20)

    peaksv = np.array([])
    for j in peaks:
        peaksv = np.append(peaksv,dfE[columname].loc[j])

    maxl = np.argmax(peaksv)

    index.append((peaks[maxl-1],peaks[maxl]))

#%%

slopes = []
cg = [50,100,200,500,1000]
j=0
for element in index:
    y = np.array(element)
    if j == 2:
        y=[y[0],y[1]-900]
    
    slope, intercept, r, p, se = stats.linregress(y, E)

    slopes.append(slope)

print(slopes)
#%%

"""
Getting the slope
"""
slope, intercept, r, p, se = stats.linregress(cg, slopes)


m = slope * 20 + intercept

#%%

dfl = pd.DataFrame()
dfl, columname = reader(file_r)
dfl = dfl.loc[20:1000]
xp = np.arange(0,len(dfl[columname]),1)

b, a = signal.butter(3, 0.05)
y = dfl[columname]
yy = signal.filtfilt(b, a, y)
dfl[columname] = yy
peaks, _ = signal.find_peaks(dfl[columname], height=20)

print(peaks)
peaksv = np.array([])
for j in peaks:
    #peaksv = np.append(peaksv,dfl[columname].loc[j])
    None
#maxl = np.argmax(peaksv)


plt.plot(xp,dfl[columname])
plt.vlines(peaks[0],0,1000)
plt.vlines(peaks[1],0,1000)
#%%

slope, intercept, r, p, se = stats.linregress(peaks, E)

fit = xp*slope + intercept

#%%
plt.vlines(peaks[0],0,40)
plt.vlines(peaks[1],0,40)
plt.hlines(5.9,0,max(xp))
plt.hlines(2.9,0,max(xp))
plt.plot(xp,fit)
# %%

fit = fit.reshape(-1,1)
print(fit)
#%%

with open(file_w, 'w',newline='') as csvfile:
    # Create a writer object
    csvwriter = csv.writer(csvfile)
    # Write the fields and rows to the file
    csvwriter.writerows(fit)

# %%
