# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 22:10:26 2023

@author: krzys
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.utils import resample

#load data
df = pd.read_csv('ChickData.csv', sep = ';')

#plot boxplot
sns.boxplot(data=df, x="weight", y="feed")

df[df['feed'] == 'meatmeal']['weight'].mean()

#calculate means
df[df['feed'] == 'casein']['weight'].mean()
df[df['feed'] == 'meatmeal']['weight'].mean()

#absolute difference in means
test_stat1 = abs(df[df['feed'] == 'casein']['weight'].mean() - 
                 df[df['feed'] == 'meatmeal']['weight'].mean())

#calculate median
df[df['feed'] == 'casein']['weight'].median()
df[df['feed'] == 'meatmeal']['weight'].median()

#absolute difference in medians
test_stat2 = abs(df[df['feed'] == 'casein']['weight'].median() -
                 df[df['feed'] == 'meatmeal']['weight'].median())

#np.random.seed(112358)

n = df['feed'].size
B = 10000
variable = df['weight']

bootstrap_samples = np.random.choice(variable, size = n * B).reshape(n, B)

boot_test_stat1 = np.zeros(B)
boot_test_stat2 = np.zeros(B)

for i in range(B):
    boot_test_stat1[i] = abs(np.mean(bootstrap_samples[0:11,i]) - 
                             np.mean(bootstrap_samples[12:22,i]))
    boot_test_stat2[i] = abs(np.median(bootstrap_samples[0:11,i]) - 
                             np.median(bootstrap_samples[12:22,i]))
    
"""
Wartosc p = ilosc statystyk boostrapowych wiekszych niz te obserwowane /
            total ilosc statstyk bootstrapowych
"""
print(np.mean(boot_test_stat1 >= test_stat1))

"""
Sposrod 10000 statystyk bootstrapowych 1048 mialo srednia wieksza niz ta
obserwowana
Brak podstaw do odrzucenia H0
"""

print(np.mean(boot_test_stat2 >= test_stat2))
