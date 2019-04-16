'''

John Heisey, M.Sc. Computational Science & Engineering
February 11, 2019

Code purpose:
This script extracts the minimalist feature set by
loading the sentiment/moment set and reducing it.
'''

import numpy as np
import pandas as pd
import csv
import sys

ticker = sys.argv[1]

featName = ticker+"_norm_feat_5min_jh_new.csv"

feat = pd.read_csv(featName, sep=',', header=0)

miniFeat = feat.iloc[:,40:]
print(miniFeat.shape)

cols = [54,55,56,57,58]

miniFeat = miniFeat.drop(miniFeat.columns[cols],axis=1)

print(miniFeat.head)
print(miniFeat.shape)

feat_csv = ticker+"_norm_feat_5min_mini.csv"

np.savetxt(feat_csv,miniFeat, delimiter=",", fmt='%1.2f')

print("Feat vec (Minimalist) generated for "+ticker+".")
