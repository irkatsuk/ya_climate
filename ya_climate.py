import pandas as pd

import ya_climate_common
from ya_climate_preprocessing import stage1

df = pd.read_csv("ya_climate.csv", sep=";", decimal=".")

df = stage1(df)
