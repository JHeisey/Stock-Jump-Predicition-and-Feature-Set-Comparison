'''

John Heisey, M.Sc. Computational Science & Engineering
February 11, 2019

Code purpose:
This script plots a year histogram of jump occurences in 2013
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


aapl = pd.read_csv("AAPL_jumps.csv", header=None)
goog = pd.read_csv("GOOG_jumps.csv", header=None)
fb = pd.read_csv("FB_jumps.csv", header=None)
intc = pd.read_csv("INTC_jumps.csv", header=None)
msft = pd.read_csv("MSFT_jumps.csv", header=None)

print(aapl.iloc[:,1].value_counts().shape)

colors = ['#cef7c3','#9fdbea','#5ba8cc','#2f639e','#1d2d70']
plt.hist([aapl.iloc[:,1],goog.iloc[:,1],fb.iloc[:,1],intc.iloc[:,1],msft.iloc[:,1]], bins=50,stacked=True, normed=True, color=colors, label=['AAPL','GOOG','FB','INTC','MSFT'])
plt.xlim(0,254)
plt.legend()
plt.ylabel("Frequency")
plt.xlabel("Days")
plt.show()