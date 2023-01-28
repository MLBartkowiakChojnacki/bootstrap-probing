# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 22:10:26 2023

@author: krzys
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#load data
df = pd.read_csv('Grupa.1.csv', sep = ';')

df_area_2021 = df['X2a']
df_area_2022 = df['X2b']
df_area_2021 = df_area_2021.to_frame()
df_area_2022 = df_area_2022.to_frame()
df_area_2021['year'] = '2021'
df_area_2022['year'] = '2022'
df_area_2021.columns = ['agr_area', 'year']
df_area_2022.columns = ['agr_area', 'year']

df_all = pd.concat([df_area_2021, df_area_2022])

#plot boxplot
sns.boxplot(data=df_all, x="agr_area", y="year")

#absolute difference in means
test_stat1 = abs(df_area_2021['agr_area'].mean() - 
                 df_area_2022['agr_area'].mean())


#absolute difference in medians
test_stat2 = abs(df_area_2021['agr_area'].median() -
                 df_area_2022['agr_area'].median())


n = df_all['year'].size
B = 10000
variable = df_all['agr_area']

bootstrap_samples = np.random.choice(variable, size = n * B).reshape(n, B)

boot_test_stat1 = np.zeros(B)
boot_test_hist1 = np.zeros(B)
boot_test_stat2 = np.zeros(B)
boot_test_hist2 = np.zeros(B)

for i in range(B):
    boot_test_hist1[i] = np.mean(bootstrap_samples[0:199,i])
    boot_test_stat1[i] = abs(np.mean(bootstrap_samples[0:199,i]) - 
                             np.mean(bootstrap_samples[200:399,i]))
    boot_test_hist2[i] = np.mean(bootstrap_samples[200:399,i])
    boot_test_stat2[i] = abs(np.median(bootstrap_samples[0:199,i]) - 
                             np.median(bootstrap_samples[200:399,i]))

fig, ax = plt.subplots()
ax.hist(boot_test_hist1, bins=20, color='c', edgecolor='k')
plt.axvline(df_area_2021['agr_area'].mean(), color='k', linestyle='dashed', linewidth=2)
plt.axvline(np.percentile(boot_test_hist1, 2.5), color='r', linestyle='dashed', linewidth=1)
plt.axvline(np.percentile(boot_test_hist1, 97.5), color='r', linestyle='dashed', linewidth=1)
plt.show()

fig, ax = plt.subplots()
ax.hist(boot_test_hist2, bins=20, color='c', edgecolor='k')
plt.axvline(df_area_2022['agr_area'].mean(), color='k', linestyle='dashed', linewidth=2)
plt.axvline(np.percentile(boot_test_hist2, 2.5), color='r', linestyle='dashed', linewidth=1)
plt.axvline(np.percentile(boot_test_hist2, 97.5), color='r', linestyle='dashed', linewidth=1)
plt.show()    

"""
Wartosc p = ilosc statystyk boostrapowych wiekszych niz te obserwowane /
            total ilosc statstyk bootstrapowych
"""
p_value = np.mean(boot_test_stat1 >= test_stat1)
print(p_value)

p_value = np.mean(boot_test_stat2 >= test_stat2)
print(p_value)


