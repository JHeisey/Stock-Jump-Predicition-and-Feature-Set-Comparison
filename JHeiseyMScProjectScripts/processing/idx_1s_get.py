'''

John Heisey, M.Sc. Computational Science & Engineering
February 11, 2019

Code purpose:
This script converts the daily event based data into
uniform 1 second intervals, and concatenates each
day of indices together
'''

import numpy as np
import pandas as pd
import csv
import glob

tickers = ["AAPL","FB","GOOG","INTC","MSFT"]


for ticker in tickers:

    print("Now working on "+ticker+": ")
    paths = "P_*/mids/mid_"+ticker+".csv"

    paths = glob.glob(paths)
    paths.sort()
    day_num = 0

    days = []
    year_idx = []
    
    prev_day_idx = 0

    for day in paths:
        day_num += 1
        print("Day "+str(day_num)+" started")
        df_day = pd.read_csv(day, header=None)

        nameMid = ticker+"_mid_full.csv"

        secPrice = 0
        secTime = 0

        cnt = 34200
        sec = 34200

        for i in range(len(df_day.iloc[:,2])):
                if df_day.iloc[i,0] == cnt and cnt <= 57600:
                        days.append(day_num)
                        year_idx.append(i+prev_day_idx)
                        cnt +=1


                if sec < df_day.iloc[i,0]:
                        secPrice = df_day.iloc[i,2]
                        secTime = df_day.iloc[i,0]
                        sec = df_day.iloc[i,0]

                if df_day.iloc[i,0] > cnt and df_day.iloc[i,0] < 57600:
                        days.append(day_num)
                        year_idx.append(i+prev_day_idx)
                        cnt +=1
                
        prev_day_idx += len(df_day.iloc[:,2])
            
            

    mids_final = zip(year_idx, days)

    print("Length of year_index: "+str(len(year_idx)))
    print("Equivalent to "+str(len(year_idx)/23400.0)+" days")
    
    csvfile = ticker+"_idx_1s_year_10oclock.csv"

    with open(csvfile, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(mids_final)

    print("CSV file for "+ticker+" written.")

