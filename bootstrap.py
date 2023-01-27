# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 20:14:41 2023

@author: krzys
"""

from scipy.stats import ttest_ind
from scipy.stats import bootstrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bootstrap, ttest_ind
from sklearn.utils import resample

df = pd.read_csv(r"C:\Users\krzys\OneDrive\Dokumenty\repo_git\bootstrap-probing\data\raw\Grupa.1.csv", delimiter = ";", decimal=",", thousands=".")

df_uprawa_2021 = df["X2a"]
df_uprawa_2022 = df["X2b"]

#Figure settings
sns.set(font_scale=1.65)
fig = plt.figure(figsize=(10,6))
fig.subplots_adjust(hspace=.5)    

#Plot top histogram
ax = fig.add_subplot(2, 1, 1)
ax = df_uprawa_2021.hist(bins=50)
plt.xlabel('Area')
plt.ylabel('Frequency')

#Plot bottom histogram
ax2 = fig.add_subplot(2, 1, 2)
ax2 = df_uprawa_2022.hist(bins=50)

plt.xlabel('Area')
plt.ylabel('Frequency')

plt.show()

print('Agriculture area in 2021: {}'.format(df_uprawa_2021.mean()))
print('Est. agriculture area in 2022: {}'.format(df_uprawa_2022.mean()))

area_2021 = []
for i in range(10000):
    area_2021.append((resample(df_uprawa_2021)))

area_2021 = np.mean(area_2021, axis=1)
area_2021

area_est_2022 = []
for i in range(10000):
    area_est_2022.append((resample(df_uprawa_2022)))
    
area_est_2022 = np.mean(area_est_2022, axis=1)
area_est_2022


differences = area_est_2022 - area_2021
lower_bound = np.percentile(differences, 2.5)
upper_bound = np.percentile(differences, 97.5)

fig = plt.figure(figsize=(10,3))
ax = plt.hist(differences, bins=30)

plt.xlabel('Difference in agriculture area')
plt.ylabel('Frequency')
plt.axvline(lower_bound, color='r')
plt.axvline(upper_bound, color='r')
plt.title('Bootstrapped Population (Difference Between 2 Groups)')
plt.show()

print('Lower bound: {}'.format(lower_bound))
print('Upper bound: {}'.format(upper_bound))

#testowanie hipotez
combined = np.concatenate((area_2021, area_est_2022), axis=0)

perms_2021 = []
perms_2022 = []

for i in range(10000):
    perms_2021.append(resample(combined, n_samples = len(df_uprawa_2021)))
    perms_2022.append(resample(combined, n_samples = len(df_uprawa_2022)))
    
dif_bootstrap_means = (np.mean(perms_2021, axis=1)-np.mean(perms_2022, axis=1))
dif_bootstrap_means

obs_difs = (np.mean(perms_2021) - np.mean(perms_2022))
print('observed difference in means: {}'.format(obs_difs))

p_value = dif_bootstrap_means[dif_bootstrap_means >= obs_difs]
print('p-value: {}'.format(p_value))

fig = plt.figure(figsize=(10,3))
ax = plt.hist(dif_bootstrap_means, bins=30)

plt.xlabel('Difference in agriculture area')
plt.ylabel('Frequency')
plt.axvline(obs_difs, color='r')
plt.title('Bootstrapped Population (Combined data)')
plt.show()

if p_value >= 0.05: print('Brak podstaw do odrzucenia H0')


#%%

import pandas as pd
import numpy as np
from bstrap import bootstrap, boostrapping_CI

# 1. implement metric
metric = np.mean

# 4. compare method 1 and 2
stats_method1, stats_method2, p_value = bootstrap(metric, df_uprawa_2021, df_uprawa_2022, nbr_runs=10000)
print(stats_method1)
print(stats_method2)
print(p_value)

# compute CI and mean for each method separately
stats_method1 = boostrapping_CI(metric, df_uprawa_2021, nbr_runs=1000)
stats_method2 = boostrapping_CI(metric, df_uprawa_2022, nbr_runs=1000)
print(stats_method1)
print(stats_method2)


#%%

data_1 = (df_uprawa_2021,)
data_2 = (df_uprawa_2022,)
res_1 = bootstrap(data_1, np.mean, confidence_level=0.95, n_resamples = 9999)
res_2 = bootstrap(data_2, np.mean, confidence_level=0.95, n_resamples = 9999)

fig, ax = plt.subplots()
ax.hist(res_1.bootstrap_distribution, bins=20, color='c', edgecolor='k')
plt.axvline(df["X2a"].mean(), color='k', linestyle='dashed', linewidth=2)
plt.axvline(res_1.confidence_interval[0], color='r', linestyle='dashed', linewidth=1)
plt.axvline(res_1.confidence_interval[1], color='r', linestyle='dashed', linewidth=1)
plt.show()


fig, ax = plt.subplots()
ax.hist(res_2.bootstrap_distribution, bins=20, color='c', edgecolor='k')
plt.axvline(df["X2b"].mean(), color='k', linestyle='dashed', linewidth=2)
plt.axvline(res_2.confidence_interval[0], color='r', linestyle='dashed', linewidth=1)
plt.axvline(res_2.confidence_interval[1], color='r', linestyle='dashed', linewidth=1)
plt.show()

#%%
