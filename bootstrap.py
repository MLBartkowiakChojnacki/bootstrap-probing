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

def my_statistic(sample1, sample2, axis):
    statistic, p_value = ttest_ind(sample1, sample2)
    return statistic

# 2. load data
df = pd.read_csv(r"C:\Users\krzys\OneDrive\Dokumenty\repo_git\bootstrap-probing\data\raw\Grupa.1.csv", delimiter = ";", decimal=",", thousands=".")

# 3. reformat data to a single pandas dataframe per method
df_uprawa_2021 = df["X2a"]
df_uprawa_2022 = df["X2b"]
data = (df_uprawa_2021, df_uprawa_2022)
res = bootstrap(data, my_statistic, n_resamples=9999, method='basic', confidence_level=0.95)


fig, ax = plt.subplots()
ax.hist(res.bootstrap_distribution, bins=20)
plt.show()

#%%
import matplotlib.pyplot as plt
from scipy.stats import bootstrap, ttest_ind
import numpy as np


df_uprawa_2021 = df["X2a"]
df_uprawa_2022 = df["X2b"]  # samples must be in a sequence
data_1 = (df_uprawa_2021,)
data_2 = (df_uprawa_2022,)
res_1 = bootstrap(data_1, np.mean, confidence_level=0.95)
res_2 = bootstrap(data_2, np.mean, confidence_level=0.95)

print(ttest_ind(res_1.bootstrap_distribution,res_2.bootstrap_distribution)[1])
a = round(3.6282382319929215e-37, 4)

#%%
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 20:40:52 2023

@author: krzys
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from bstrap import bootstrap

# Create a dataframe
dataframe = pd.DataFrame({'Fertilizer': np.repeat(['daily', 'weekly'], 15),
						'Watering': np.repeat(['daily', 'weekly'], 15),
						'height': [14, 16, 15, 15, 16, 13, 12, 11,
									14, 15, 16, 16, 17, 18, 14, 13,
									14, 14, 14, 15, 16, 16, 17, 18,
									14, 13, 14, 14, 14, 15]})


# Performing two-way ANOVA
model = ols('height ~ C(Fertilizer) + C(Watering) + C(Fertilizer):C(Watering)',
			data=dataframe).fit()
result = sm.stats.anova_lm(model, type=2)

# Print the result
print(result)



#%%

