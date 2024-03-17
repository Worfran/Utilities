#%%

import numpy as np
import pandas as pd


#%%

# Medias
seed_mu = [11.3, 9.5, 12.2]

d = np.abs(np.random.normal(0.1, 0.01, 9))

means =[]

for i in range (0,3):
    for j in range(0,3):
        means.append(seed_mu[i%3] + j%3 * d[i+j])


#%%
# Varianzas
d = np.abs(np.random.normal(0.1, 0.01, 9))
seed_sig = [3.9, 11.3, 2.7]

sigs = []

for i in range (0,3):
    for j in range(0,3):
        sigs.append(seed_sig[i%3] + j%3 * d[i+j])


#%%
        
print(sigs)
print(means)

#%%

# Generate 9 subsets
subsets = [np.abs(np.random.normal(mu, sigma, 9)) for mu, sigma in zip(means, sigs)]

# Print the subsets
for i, subset in enumerate(subsets, 1):
    print(f"Subset {i}: {subset}")



# %%
    
#Data frame
    
sleep_hours = ["Menos de 6", "6-8 horas", "mas de 8 horas"]
age_groups = ["17-20", "21-25", "26-30"]

dft = pd.DataFrame()

# Create and export dataframes
for i, subset in enumerate(subsets):
    # Create a dataframe
    df = pd.DataFrame(subset, columns=['Subset'])
    
    # Add the sleep hours and age group columns
    df['Horas de sueño'] = sleep_hours[i % 3]
    df['Grupo de edad'] = age_groups[i // 3]

    dft = pd.concat([dft,df])

dft.reset_index(drop=True, inplace=True)
print(dft)
    # %%


dft.to_csv('Proba_2_p1.csv', index=False)

# %%
