'''

John Heisey, M.Sc. Computational Science & Engineering
February 11, 2019

Code purpose:
This script plots a LOB snapshot example where the
moment features could prove useful
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

tenAskPrices = np.array([50.02,50.04,50.06,50.08,50.09,50.10,50.11,50.12,50.13,50.14])
tenAskVols = np.array([2,2,3,4,12,27,28,37,43,50])
askSumStats = np.repeat(tenAskPrices,tenAskVols)


maxVol = max(tenAskVols)

tenBidPrices = np.array([49.98, 49.96, 49.94, 49.92, 49.91, 49.90, 49.89, 49.88, 49.87, 49.86])-0.15
tenBidVols = np.array([3,2,5,3,11,25,35,30,43,40])
bidSumStats = np.repeat(tenBidPrices,tenBidVols)

mid = (tenAskPrices[0] + tenBidPrices[0])/2


minPrice = outsideBids[-1]-0.01
maxPrice = outsideAsks[-1]+0.01


plt.figure()
plt.title("Skewed LOB State")
plt.ticklabel_format(useOffset=False)
plt.barh(tenAskPrices,tenAskVols,0.01,color='#55cc63', label="Ask")

plt.barh(tenBidPrices,tenBidVols,0.01,color='#d64242', label="Bid")

plt.axhline(mid, linestyle='--', linewidth=2, color='k',label="Midprice")
plt.ylim(minPrice, maxPrice)
plt.xlim(0,maxVol*1.4)
plt.xlabel("Volume")
plt.ylabel("Price ($)")
plt.legend()

plt.show()
