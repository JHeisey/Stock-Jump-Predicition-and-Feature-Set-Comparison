'''

John Heisey, M.Sc. Computational Science & Engineering
February 11, 2019

Code purpose:
This script plots nested pie charts of the classified
jump distributions between all 5 stocks
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


aapl = pd.read_csv("AAPL_jumps.csv", header=None)
goog = pd.read_csv("GOOG_jumps.csv", header=None)
fb = pd.read_csv("FB_jumps.csv", header=None)
intc = pd.read_csv("INTC_jumps.csv", header=None)
msft = pd.read_csv("MSFT_jumps.csv", header=None)

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

print(aapl)


all_days = np.concatenate((aapl_days.values, goog_days.values, fb_days.values, intc_days.values, msft_days.values))
all_L = np.concatenate((aapl_L.values, goog_L.values, fb_L.values, intc_L.values, msft_L.values))

all_L_pos = all_L[all_L > 0]
all_L_neg = all_L[all_L < 0]

print("All neg:", np.shape(all_L_neg))
print("All pos:", np.shape(all_L_pos))
print("Avg. neg:", np.mean(all_L_neg))
print("Avg. pos:", np.mean(all_L_pos))

print("\% neg:", float(len(all_L_neg))/(len(all_L_neg) + len(all_L_pos)))
print("\% pos:", float(len(all_L_pos))/(len(all_L_neg) + len(all_L_pos)))


fig, axes = plt.subplots(nrows=2, sharex=True)
plt.xlim(5,40)
# axes[1].bar(y, x1, align='center', color='gray')
axes[0].hist(all_L_pos[all_L_pos < 100], bins=40, color="#55cc63", edgecolor='white')
# axes[0].bar(y, x2, align='center', color='gray')
axes[1].hist(-1*all_L_neg[all_L_neg > -100], bins=60, color='#d64242', edgecolor='white')
axes[1].invert_yaxis()
axes[0].set_ylabel('Positive Counts')
axes[1].set_ylabel('Negative Counts')
plt.xlabel(r"Barndorff-Nielsen Jump Statistic Magnitude")
axes[0].set_title("Distribution of B-N Classified Jumps (All Stocks)")
# axes[1].invert_xaxis()

plt.show()

print(len(aapl_L[aapl_L > 0]))

plt.rcParams['font.size'] = 9.0

group_names=['Positive', 'Negative']
group_size=[len(all_L_pos),len(all_L_neg)] #all pos/neg counts
subgroup_names=['AAPL', 'GOOG', 'FB', 'MSFT', 'INTC','INTC', 'MSFT', 'FB', 'GOOG', 'AAPL']
subgroup_size=[len(aapl_L[aapl_L > 0]),
			   len(goog_L[goog_L > 0]),
			   len(fb_L[fb_L > 0]),
			   len(msft_L[msft_L > 0]),
			   len(intc_L[intc_L > 0]),
			   len(intc_L[intc_L < 0]),
			   len(msft_L[msft_L < 0]),
			   len(fb_L[fb_L < 0]),
			   len(goog_L[goog_L < 0]),
			   len(aapl_L[aapl_L < 0])]
 

# Create colors
b, a=[plt.cm.Reds, plt.cm.Greens]
 
# First Ring (outside)
fig, ax = plt.subplots()
ax.axis('equal')
mypie, _ = ax.pie(group_size, radius=1.2, labels=group_names, colors=[a(0.45), b(0.45)], startangle=90)
plt.setp( mypie, width=0.3, edgecolor='white')
 
# Second Ring (Inside)
mypie2, _ = ax.pie(subgroup_size, radius=1.3-0.3, labels=subgroup_names, labeldistance=0.8, colors=[a(0.7), a(0.6), a(0.5), a(0.4), a(0.3), b(0.3), b(0.4), b(0.5), b(0.6), b(0.7)], startangle=90)
plt.setp( mypie2, width=0.4, edgecolor='white')

plt.title("Jump Distribution (Pos/Neg) for Individual Stocks (30s)")

plt.show()
