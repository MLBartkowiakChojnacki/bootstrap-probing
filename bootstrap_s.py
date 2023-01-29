import pandas as pd
import numpy as np
from bstrap import bootstrap
import seaborn as sns
import matplotlib.pyplot as plt

#ladowanie danych
df = pd.read_csv(r'data/raw/Grupa.1.csv', sep = ';')
df_regional_structure = pd.read_csv(r"data/raw/Regional.Structure.csv", sep = ";")
df_region_names = pd.read_csv(r'data/raw/REGION_NAMES.csv', sep = ";")
df_region_names.columns = ['RegionC', 'Prov']
df = df.merge(df_regional_structure, on = 'RegionA', how = 'left')
df = df.merge(df_region_names, on = 'RegionC', how = 'left')
province = df['Prov'].unique().tolist()
 
#kombinacje bez powtorzen dla dwojek wojewodztw 
province_combinations = [(a, b) for idx, a in enumerate(province) for b in province[idx + 1:]]

#metryka, ktora bedziemy badac
metric = np.mean

#badania istotnosci roznic miedzy deklarowana wielkoscia uprawy
#dla 2022 roku miedzy wojewodztwami
prov_1 = []
prov_2 = []
avg_1 = []
avg_2 = []
conf_inter_1 = []
conf_inter_2 = []
p_value_list = []

for prov in province_combinations:

    df_uprawa_2022_1 = df["X2b"][df['Prov'] == prov[0]]
    df_uprawa_2022_2 = df["X2b"][df['Prov'] == prov[1]]
    
    stats_method1_22, stats_method2_22, p_value_22 = bootstrap(metric, df_uprawa_2022_1, df_uprawa_2022_2, nbr_runs=10000)  
    prov_1.append(prov[0])
    prov_2.append(prov[1])
    avg_1.append(stats_method1_22['avg_metric'])
    avg_2.append(stats_method2_22['avg_metric'])
    conf_inter_1.append((round(stats_method1_22['metric_ci_lb'],2),
                         round(stats_method1_22['metric_ci_ub'],2)))
    conf_inter_2.append((round(stats_method2_22['metric_ci_lb'],2),
                         round(stats_method2_22['metric_ci_ub'],2)))
    p_value_list.append(p_value_22)

df_sum_prov_comb = pd.DataFrame(data = {'YR-YR': '2022-2022',
                                      'PROV_1': prov_1,
                                      'PROV_2': prov_2,
                                      'AVG_AREA_1': avg_1,
                                      'CI_1': conf_inter_1,
                                      'AVG_AREA_2': avg_2,
                                      'CI_2': conf_inter_2,
                                      'P_VALUE': p_value_list})
    

#badanie istotnosci roznic miedzy wielkoscia uprawy w 2021 a deklarowana
#wielkoscia uprawy a 2022 dla wojewodztw
prov_1 = []
prov_2 = []
avg_1 = []
avg_2 = []
conf_inter_1 = []
conf_inter_2 = []
p_value_list = []

for prov in province:

    df_uprawa_2021 = df["X2a"][df['Prov'] == prov]
    df_uprawa_2022 = df["X2b"][df['Prov'] == prov]
    
    stats_method1_21, stats_method2_22, p_value = bootstrap(metric, df_uprawa_2021, df_uprawa_2022, nbr_runs=10000)  
    prov_1.append(prov)
    prov_2.append(prov)
    avg_1.append(stats_method1_21['avg_metric'])
    avg_2.append(stats_method2_22['avg_metric'])
    conf_inter_1.append((round(stats_method1_21['metric_ci_lb'],2),
                         round(stats_method1_21['metric_ci_ub'],2)))
    conf_inter_2.append((round(stats_method2_22['metric_ci_lb'],2),
                         round(stats_method2_22['metric_ci_ub'],2)))
    p_value_list.append(p_value)

df_sum_yr_delta = pd.DataFrame(data = {'YR-YR': '2021-2022',
                                      'PROV_1': prov_1,
                                      'PROV_2': prov_2,
                                      'AVG_AREA_1': avg_1,
                                      'CI_1': conf_inter_1,
                                      'AVG_AREA_2': avg_2,
                                      'CI_2': conf_inter_2,
                                      'P_VALUE': p_value_list})
