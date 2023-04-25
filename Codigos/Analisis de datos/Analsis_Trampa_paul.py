#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
#%%
sns.set_style('darkgrid')

bbox = dict(boxstyle ="round",facecolor='white')

__Fpath__="../../"

#%%
# Data for plotting
x = np.arange(35.50, 60.00, 0.01)

m1 = 4.9015
b1 = -69.7699

m2 = 4.51930
b2 = -103.12243

y1 =m1*x + b1

y2 = m2*x + b2
#%%
plt.plot(x, y1,label="Limite superior", color = "#322AD5")
plt.plot(x, y2, label ="Limute inferior", color = "#D5322A")
  
# Assighning plot attributes
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Voltaje (V)")
plt.title('Region de estabilidad')

  
# Filling sign wave curv with cyan color
plt.fill_between(x, y1, y2, color='#3BC4B3', alpha=.5, label="Region de estabilidad")
plt.legend()
plt.savefig(__Fpath__+"Images/Region_estabilidad.png",dpi=600)

#%%