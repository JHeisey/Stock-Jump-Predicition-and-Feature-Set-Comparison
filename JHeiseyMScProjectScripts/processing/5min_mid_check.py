'''

John Heisey, M.Sc. Computational Science & Engineering
February 11, 2019

Code purpose:
This script iterates through each day of midprices, and removes
duplicate sampling, retaining the continuous nature of the
time series data
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

ticker = sys.argv[1]

idxName = ticker+"_mid_5min_year.csv"

idx = pd.read_csv(idxName, header=None)

unique_df = idx[idx[2]==1].drop_duplicates(subset=0,keep='first')

for i in range(2,253):
	df_tmp = idx[idx[2]==i].drop_duplicates(subset=0,keep='first')
	unique_df = unique_df.append(df_tmp,ignore_index=True)

print(unique_df)
plt.plot(unique_df.iloc[:,0])
plt.show()


midName = ticker+"_mid_5min_unique.csv"

unique_df.to_csv(midName, header=False, index=False)
