'''

John Heisey, M.Sc. Computational Science & Engineering
February 11, 2019

Code purpose:
This script classifies midprice jump data using the
method proposed by Lee & Mykland (2008) for the 30 second
time series data
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import sys
ticker = sys.argv[1]

#nameMid = "30s_stocks/"+ticker+"_mid_30_unique.csv"
nameMid = ticker+"_mid_30_year_unique.csv"

mid = pd.read_csv(nameMid, sep=',', header=None)



S = mid.ix[:,1]
midTime = mid.ix[:,0]
midDay = mid.ix[:,2]	

# print midTime
print(midTime.head())
K = 1560
a = 0.1
n = 780

print("Jump testing for K="+str(K)+", N="+str(n)+", and alpha="+str(a)+".")

def real_bi_var(S,j,i):
	S_0 = S[j+2:i].values
	S_1 = S[j+1:i-1].values
	S_2 = S[j:i-2].values
	rbv = 1.0/(i-j)*np.sum(abs(np.log(S_0/S_1))*abs(np.log(S_1/S_2)))
	return rbv
	
	

def test_for_jumps(S,i,K,B_2):
	sig_sqr = real_bi_var(S, i-K, i)
	L = np.log(S[i]/S[i-1])/np.sqrt(sig_sqr)
	if abs(L)>=B_2:
		print(abs(L))
	if abs(L)>=B_2 and midTime[i] == 34200:
		print("jump happened at 9:30")
	return L


def find_all_jumps(S,K,a,n):
	B_1 = -np.log(-np.log(1.0-a))
	print("B_1 is: "+str(B_1))
	c = np.sqrt(2.0)/np.pi
	C_n =np.sqrt(2.0*np.log(n))/c - (np.log(np.pi) + np.log(np.log(n)))/(2.0*c*np.sqrt(2.0*np.log(n)))
	S_n = 1.0/(c*(np.sqrt(2.0*np.log(n))))
	B_2 = B_1 * S_n + C_n
	print("B_2 is: "+str(B_2))
	J = []
	jump_stats = []
	days = []
	secs = []
	prices = []

	for i in range(K,len(S)):
		L = test_for_jumps(S,i,K,B_2)
		if abs(L) >= B_2 and midTime[i] != 34200:
			# print "intraday jump happened"
			J.append(i)
			jump_stats.append(L)
			days.append(midDay[i])
			secs.append(midTime[i])
			prices.append(S[i])
			# print "Num of jumps so far: "+str(len(J))


	return J, jump_stats, days, secs, prices

jump_idx, jump_stats, days, secs, prices = find_all_jumps(S, K, a, n)


'''
print jump_idx
print len(jump_idx)

print jump_stats
print len(jump_stats)

print days
print len(days)

print secs
print len(secs)

print prices
print len(prices)
'''
#plt.hist(secs, bins=30)
#plt.show()

print("percentage of events with jumps: "+str(len(jump_stats)*100.0/len(S))+"%.")
print("Average jumps per day: "+str(len(jump_stats)/252.0)+".")

#csvfile = "jumps/"+ticker+"_jumps.csv"
csvfile = ticker+"_jumps.csv"


info = [secs,days,prices,jump_idx,jump_stats]

with open(csvfile, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(list(zip(*info)))



