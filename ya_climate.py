import pandas as pd

import ya_climate_common
from ya_climate_preprocessing import stage1
from ya_calculation import research
from ya_hypotheses import hypotheses
from ya_regression import ya_regression

df = pd.read_csv("ya_climate.csv", sep=";", decimal=".")

df = stage1(df)

ya_climate_common.check_data(df)

df, avg_age, med_type_of_cool, avg_comf_temp_age, avg_temp_rh, list_of_corr = research(df)
sorted_list = sorted(list_of_corr, key=lambda x: -abs(x[0]))
for el in sorted_list:
    print(el)
hypotheses(df)
ya_regression(df)