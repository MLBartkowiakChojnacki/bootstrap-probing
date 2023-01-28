import pandas as pd
import numpy as np
from bstrap import bootstrap
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#load data and add info about regions and vovoiderships
df = pd.read_csv('Grupa.1.csv', sep = ';')
df_regional_structure = pd.read_csv("Regional.Structure.csv", sep = ";")
df_region_names = pd.read_csv('REGION_NAMES.csv', sep = ";")
df_region_names.columns = ['RegionC', 'Woj']
df = df.merge(df_regional_structure, on = 'RegionA', how = 'left')
df = df.merge(df_region_names, on = 'RegionC', how = 'left')
woj = df['Woj'].unique().tolist()
  
woj_combinations = [(a, b) for idx, a in enumerate(woj) for b in woj[idx + 1:]]

metric = np.mean

woj_1 = []
woj_2 = []
avg_1 = []
avg_2 = []
conf_inter_1 = []
conf_inter_2 = []
p_value = []

for woj in woj_combinations:

    df_uprawa_2022_1 = df["X2b"][df['Woj'] == woj[0]]
    df_uprawa_2022_2 = df["X2b"][df['Woj'] == woj[1]]
    
    stats_method1_22, stats_method2_22, p_value_22 = bootstrap(metric, df_uprawa_2022_1, df_uprawa_2022_2, nbr_runs=1000)  
    woj_1.append(woj[0])
    woj_2.append(woj[1])
    avg_1.append(stats_method1_22['avg_metric'])
    avg_2.append(stats_method2_22['avg_metric'])
    conf_inter_1.append((round(stats_method1_22['metric_ci_lb'],2),
                         round(stats_method1_22['metric_ci_ub'],2)))
    conf_inter_2.append((round(stats_method2_22['metric_ci_lb'],2),
                         round(stats_method2_22['metric_ci_ub'],2)))
    p_value.append(p_value_22)

df_summary = pd.DataFrame(data = {'WOJ_1': woj_1,
                                  'WOJ_2': woj_2,
                                  'AVG_AREA_2022_1': avg_1,
                                  'CONF_INT_2022_1': conf_inter_1,
                                  'AVG_AREA_2022_2': avg_2,
                                  'CONF_INT_2022_2': conf_inter_2,
                                  'P_VALUE': p_value})