'''

John Heisey, M.Sc. Computational Science & Engineering
February 11, 2019

Code purpose:
This script agglomerates all daily midprice data into
one continuous time series
'''

import numpy as np
import pandas as pd
import glob

tickers = ["AAPL","FB","GOOG","INTC","MSFT"]

for ticker in tickers:

    paths = "P_*/mids/mid_"+ticker+".csv"

    paths = glob.glob(paths)
    paths.sort()
    #print(paths)

    df_tot = pd.read_csv(paths[0], header=None)

    paths = paths[1:]

    for i in paths:
        print(i)
        df_tmp = pd.read_csv(i, header=None)
        df_tot = df_tot.append(df_tmp, ignore_index=True)

    print(df_tot.count)

    df_tot.to_csv(ticker+"_mid_full.csv")
    print("Completed for ticker: "+ticker)
