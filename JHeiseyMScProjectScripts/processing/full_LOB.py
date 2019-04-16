'''

John Heisey, M.Sc. Computational Science & Engineering
February 11, 2019

Code purpose:
This script agglomerates all daily LOB data into
one continuous time series
'''

import numpy as np
import pandas as pd
import glob

tickers = ["AAPL","FB","GOOG","INTC","MSFT"]

for ticker in tickers:
    paths = "P_*/books/books_"+ticker+".txt"

    paths = glob.glob(paths)
    paths.sort()
print("starting ticker: "+ticker)

df_tot = pd.read_csv(paths[0], header=None)

paths = paths[1:]

for i in paths:
    print(i)
    df_tmp = pd.read_csv(i, header=None)
    df_tot = df_tot.append(df_tmp, ignore_index=True)

print(df_tot.count)

df_tot.to_csv(ticker+"_books_full.csv")
print("Complete.\n")
