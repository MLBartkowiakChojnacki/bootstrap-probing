# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 20:14:41 2023

@author: krzys
"""

import pandas as pd
import numpy as np
from bstrap import bootstrap, boostrapping_CI

# 1. implement metric
metric = np.mean

# 2. load data
df = pd.read_csv(r"C:\Users\krzys\OneDrive\Dokumenty\repo_git\bootstrap-probing\data\raw\Grupa.1.csv", delimiter = ";", decimal=",", thousands=".")

# 3. reformat data to a single pandas dataframe per method
df_uprawa_2021 = df["X2a"]
df_uprawa_2022 = df["X2b"]

# 4. compare method 1 and 2
stats_method1, stats_method2, p_value = bootstrap(metric, df_uprawa_2021, df_uprawa_2022, nbr_runs=4000)
print(stats_method1)
print(stats_method2)
print(p_value)

#%%
from scipy.stats import ttest_ind
from scipy.stats import bootstrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def my_statistic(sample1, sample2, axis):
    statistic, p_value = ttest_ind(sample1, sample2)
    return statistic

# 2. load data
df = pd.read_csv(r"C:\Users\krzys\OneDrive\Dokumenty\repo_git\bootstrap-probing\data\raw\Grupa.1.csv", delimiter = ";", decimal=",", thousands=".")

# 3. reformat data to a single pandas dataframe per method
df_uprawa_2021 = df["X2a"]
df_uprawa_2022 = df["X2b"]
data = (df_uprawa_2021, df_uprawa_2022)
res = bootstrap(data, my_statistic, n_resamples=9999, method='basic')


fig, ax = plt.subplots()
ax.hist(res.bootstrap_distribution, bins=100)
plt.show()

#%%
