'''

John Heisey, M.Sc. Computational Science & Engineering
February 11, 2019

Code purpose:
This script takes the feature, midprice, and jump input,
and classifies each timestep as possessing a jump or not.
There is also an option to oversample the jumps to reduce
the data imbalance (for no oversampling set n_over = 1)
'''

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import sys

ticker = sys.argv[1]
print("Classifying jumps for:",ticker)
featName = ticker+"_features_kz.csv"
jumpName = "jumps/"+ticker+"_5min_jump_idx.csv"
mid30Name = "5min_stocks/"+ticker+"_mid_5min_unique.csv"

feat = pd.read_csv(featName, header=None)
jump = pd.read_csv(jumpName, header=None)
mid30 = pd.read_csv(mid30Name, header=None)

n_over = 50
total_feat = mid30.shape[0] + jump.shape[0]*(n_over - 1)


jump_idx = jump.iloc[:,3].values
jump_col = np.zeros(mid30.shape[0])
jump_col[jump_idx] = 1


mid30['3'] = pd.Series(jump_col, index=mid30.index)


jump_class = np.zeros((total_feat,feat.shape[1]-1))


t0 = time.time()

feat_set = []

jumps_found = 0

j_st = 0

#changed  mid30.iloc[i,2] > 2.0 from >= 2.0

for i in range(mid30.shape[0]):
        #print(mid30.iloc[i,0],mid30.iloc[i,2])
        if (mid30.iloc[i,2] > 2.0):# and (mid30.iloc[i,2] < 9.0):
                #for j in range((i-1)*29,min(i*31,feat.shape[0])):
                for j in range(j_st,feat.shape[0]):
                        if (j > j_st + 20000) and (j_st > 0):
                                print("Exceeded 10000 steps to search")
                                break



                        if (mid30.iloc[i,2] == feat.iloc[j,-1]) and (mid30.iloc[i,0] == feat.iloc[j,-2]):
                                print(feat.iloc[j,-2],feat.iloc[j,-1])
                                print(i*300,j)
                                j_st = j+1
                                try:
                                    if mid30.iloc[i+1,3] == 1:
                                            print("jump found @")
                                            print(mid30.iloc[i,2],mid30.iloc[i,0])
                                            #break
                                            jumps_found += 1
                                            for k in range(j-n_over,j):
                                                feature = feat.iloc[k,:].values.tolist()
                                                feature.append(1)
                                                feat_set.append(feature)
                                except:
                                    print("Exception error")
                                else:
                                    feature = feat.iloc[j,:].values.tolist()
                                    feature.append(0)
                                    feat_set.append(feature)
                                break


t1 = time.time()



import csv

with open(ticker+"_RNN_features_5min_kz_shiftback.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(feat_set)




print("time taken:"+str(t1-t0))
#print(feat.head())
#print(mid30.head())
#print(jump.head())
print("Jumps found: "+str(jumps_found))
print("Jumps in feat vec: "+str(jumps_found*n_over))
print(len(feat_set))
#print(feat_set)
