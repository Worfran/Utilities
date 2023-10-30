#%%

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

#%%
"""
Settings
"""
sns.set_style('darkgrid')

plt.rcParams["font.family"] = "serif"

plt.rcParams["font.size"] = "16"

bbox = dict(boxstyle ="round",facecolor='white')

__Fpath__="../../"

sufijo = "Data/Proyecto experimental/Frank_Majo_Proy_Exp/Grafeno/"

#%%

"""
Set the file
"""
url ="100nm/50kV-1uA-5sLiveTime-Grafeno100nm.Spe"

file = __Fpath__+sufijo+url

#%%

"""
Reading 
"""


df = pd.read_csv(file, delimiter=" ",  
                 header=None, error_bad_lines=False)

#%%