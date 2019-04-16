'''

John Heisey, M.Sc. Computational Science & Engineering
February 11, 2019

Code purpose:
This script normalizes the extracted feature set for
neural network training purposes
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import scipy.stats
import sys

ticker = sys.argv[1]

featName = ticker+"_RNN_features_5min_shiftback.csv"

feat = pd.read_csv(featName,header=None)

np.set_printoptions(precision=2, suppress=True)
#print(feat.iloc[:,-12:])
print("Features loaded..")

feat = feat.fillna(0)
norm_feat = np.zeros((feat.shape[0]-240,feat.shape[1]-2))

print("Normalizing..")
#loop to normalize all LOB feature data
for i in range(240,feat.shape[0]):
    for j in range(95):
        #print(i,j)
        norm_feat[i-240,j] = scipy.stats.zscore(feat.iloc[i-240:i,j])[-1]


#loop to normalize all sent feature data
for j in range(95,feat.shape[1]-3):
    norm_feat[:,j] = scipy.stats.zscore(feat.iloc[240:,j])

norm_feat[:,-1] = feat.iloc[240:,-1]
print(norm_feat[:,-12:])
print("Saving..")

norm_feat = np.nan_to_num(norm_feat)

norm_feat_csv = ticker+"_norm_feat_5min_jh_shiftback.csv"
np.savetxt(norm_feat_csv,norm_feat, delimiter=",", fmt='%1.3f')

print("Finished for:",ticker)
