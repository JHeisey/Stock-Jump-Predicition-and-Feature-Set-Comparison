'''

John Heisey, M.Sc. Computational Science & Engineering
February 11, 2019

Code purpose:
This script plots intraday jump time distributions
of all 5 stocks
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#aapl = pd.read_csv("AAPL_jumps.csv", header=None)
#goog = pd.read_csv("GOOG_jumps.csv", header=None)
#fb = pd.read_csv("FB_jumps.csv", header=None)
#intc = pd.read_csv("INTC_jumps.csv", header=None)
#msft = pd.read_csv("MSFT_jumps.csv", header=None)

aapl = pd.read_csv("AAPL_jumps_nomorn.csv", header=None)
goog = pd.read_csv("GOOG_jumps_nomorn.csv", header=None)
fb = pd.read_csv("FB_jumps_nomorn.csv", header=None)
intc = pd.read_csv("INTC_jumps_nomorn.csv", header=None)
msft = pd.read_csv("MSFT_jumps_nomorn.csv", header=None)

aapl_days = aapl.iloc[:,1]
goog_days = goog.iloc[:,1]
fb_days = fb.iloc[:,1]
intc_days = intc.iloc[:,1]
msft_days = msft.iloc[:,1]

aapl_L = aapl.iloc[:,4]
goog_L = goog.iloc[:,4]
fb_L = fb.iloc[:,4]
intc_L = intc.iloc[:,4]
msft_L = msft.iloc[:,4]

aapl_t = aapl.iloc[:,0]
goog_t = goog.iloc[:,0]
fb_t = fb.iloc[:,0]
intc_t = intc.iloc[:,0]
msft_t = msft.iloc[:,0]

fig, axes = plt.subplots(nrows=5, sharex=True)
plt.xlim(9.2,16.3)
N = 20
plt.ylim(0,200)
axes[0].hist(aapl_t/3600., bins=N, color='#cef7c3', edgecolor=None)
axes[1].hist(goog_t/3600., bins=N, color='#9fdbea', edgecolor=None)
axes[2].hist(fb_t/3600., bins=N, color='#5ba8cc', edgecolor=None)
axes[3].hist(intc_t/3600., bins=N, color='#2f639e', edgecolor=None)
axes[4].hist(msft_t/3600., bins=N, color='#1d2d70', edgecolor=None)
axes[0].set_title("Intraday Jump Time Distributions")
axes[0].set_ylabel("AAPL")
axes[1].set_ylabel("GOOG")
axes[2].set_ylabel("FB")
axes[3].set_ylabel("INTC")
axes[4].set_ylabel("MSFT")

plt.xlabel("Time of Day (24hr)")
plt.show()
