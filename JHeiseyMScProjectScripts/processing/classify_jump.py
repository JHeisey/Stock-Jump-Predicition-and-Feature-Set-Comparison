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
featName = ticker+"_features_jh_new.csv"
#featName = ticker+"_features_no142.csv"
#jumpName = "jumps/"+ticker+"_jumps.csv"
jumpName = ticker+"_jumps.csv"
#mid30Name = "30s_stocks/"+ticker+"_mid_30_year.csv"
mid30Name = ticker+"_mid_30_unique.csv"
#mid30Name = ticker+"_mid_30_year_unique.csv"


feat = pd.read_csv(featName, header=None)
jump = pd.read_csv(jumpName, header=None)
mid30 = pd.read_csv(mid30Name, header=None)

n_over = 15
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
j_prev= 0

#changed  mid30.iloc[i,2] > 2.0 from >= 2.0
#changed j_st = j+1 to = j

for i in range(mid30.shape[0]):
        #t1 = time.time()
        #if t1 - t0 > 1200:
        #        break
        print("checking for this mid time: ",mid30.iloc[i,2],mid30.iloc[i,0])
        #print(mid30.iloc[i,0],mid30.iloc[i,2])
        if (mid30.iloc[i,2] > 2.0):# and (mid30.iloc[i,2] < 9.0):
                #for j in range((i-1)*29,min(i*31,feat.shape[0])):
                for j in range(j_st,feat.shape[0]):
                            #print("surpassed j_st at "+str(feat.iloc[j,-2])+","+str(feat.iloc[j,-1]))
                            #j_st = 
                            #print("next 10 j's:")
                            #for y in range(j,j+11):
                                #print(feat.iloc[y,-2],feat.iloc[y,-1])
                                #print(i*30,y)
                        if (j > j_st + 10000) and (j_st > 0):
                            print("Exceeded 10000 steps to search")
                            break

                        if (mid30.iloc[i,2] == feat.iloc[j,-1]) and (mid30.iloc[i,0] == feat.iloc[j,-2]):
                                print(feat.iloc[j,-2],feat.iloc[j,-1])
                                print(i*30,j)
                                j_st = j+1
                                j_prev = j_st
                                if mid30.iloc[i,3] == 1:
                                        print("jump found @")
                                        print(mid30.iloc[i,2],mid30.iloc[i,0])
                                        #break
                                        jumps_found += 1
                                        for k in range(j-n_over,j):
                                            feature = feat.iloc[k,:].values.tolist()
                                            feature.append(1)
                                            feat_set.append(feature)

                                else:
                                    feature = feat.iloc[j,:].values.tolist()
                                    feature.append(0)
                                    feat_set.append(feature)
                                break
                        '''
                        if (mid30.iloc[i,2] == 142.0) and (mid30.iloc[i,0] == 34352.0):
                                for k in range(j,j+200):
                                    print("feature times once mid hits 142, 34352.")
                                    print(feat.iloc[j,-1],feat.iloc[j,-2])
                        if mid30.iloc[i,2] > 143.0
                        '''                

t1 = time.time()

import csv

with open(ticker+"_RNN_features_jh_new.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(feat_set)

print("time taken:"+str(t1-t0))
#print(feat.head())
#print(mid30.head())
#print(jump.head())
print("Jumps found: "+str(jumps_found))
print("Jumps in feat vec: "+str(jumps_found*15))
print(len(feat_set))
#print(feat_set)
