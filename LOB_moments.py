import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ticker = "AMZN"

lvl = 10

nameBook = 'AMZN_2012-06-21_34200000_57600000_orderbook_10.csv'
nameMess = 'AMZN_2012-06-21_34200000_57600000_message_10.csv'

demoDate = np.array([2012,6,21])

mess = pd.read_csv(nameMess, sep=',', header=None)

startTrad = 9.5*60*60
endTrad = 16*60*60

tradeHaltIdx = mess.loc[mess[1] == 7]

if not tradeHaltIdx.empty:
	print "Data contains trading halt(s), occuring at these time indices:"
	print tradeHaltIdx
else:
	print "No detected trading halts."

minutes = 5

freq = minutes*60

bounds = np.arange(startTrad,endTrad+freq, freq)

bl = np.size(bounds)

boundIdx = np.zeros(bl)

k1 = 0

k2 = np.arange(np.shape(mess)[0])

for i in k2:
	if mess.loc[i,0] >= bounds[k1]:
		boundIdx[k1] = i
		k1 += 1

boundIdx[0:-2] = boundIdx[0:-2]-1
boundIdx[-1] = np.shape(mess)[0]


tradesInfo = np.zeros([bl-1,4])

for i in range(bl-1):
	temp = mess.loc[boundIdx[i]+1:boundIdx[i+1],[1,3]]
	
	tempVis = temp[3].loc[temp[1].isin([4,2])]
	tempHid = temp[3].loc[temp[1].isin([5,2])]

	tradesInfo[i,:] = [ np.size(tempVis), np.sum(tempVis), np.size(tempHid), np.sum(tempHid)]


plt.figure()
fnt = 12
plt.bar(range(np.size(tradesInfo[:,0])),tradesInfo[:,0], color='red')
plt.bar(range(np.size(tradesInfo[:,2])),-tradesInfo[:,2], color='blue')
plt.ylabel("Number of Executions", fontsize=fnt)
plt.xlabel("Interval (%d min)" %minutes, fontsize=fnt)
plt.title("Number of Executions per %d min Interval (AMZN 2012-Jun-21)" %minutes, fontsize=fnt)

plt.show()

book = pd.read_csv(nameBook, sep=',', header=None)
book.loc[:,0:4*lvl:2] /= 10000.

eventIdx = np.random.randint(len(book))

askPricePos = np.arange(0,lvl*4,4)
askVolPos = askPricePos+1
bidPricePos = askPricePos+2
bidVolPos = bidPricePos+1
vol = np.arange(0,lvl*4,2)

maxPrice = book.loc[eventIdx,askPricePos[lvl-1]] + 0.01
minPrice = book.loc[eventIdx,bidPricePos[lvl-1]] - 0.01

maxVol = max(book.loc[eventIdx,vol])

mid = 0.5*(sum(book.loc[eventIdx,[0,2]]))

plt.figure()
plt.title("LOB Snapshot @ %d seconds" %mess.loc[eventIdx,0])
plt.ticklabel_format(useOffset=False)
plt.barh(book.loc[eventIdx,askPricePos],book.loc[eventIdx,askVolPos],0.01,color='green', label="Ask")
plt.barh(book.loc[eventIdx,bidPricePos],book.loc[eventIdx,bidVolPos],0.01,color='red', label="Bid")
plt.plot(40, mid, '<k', markersize=12, label="Mid")
plt.ylim(minPrice, maxPrice)
plt.xlim(0,maxVol)
plt.xlabel("Volume")
plt.ylabel("Price ($)")
plt.legend()

plt.show()

bookVolAsk = book.loc[eventIdx,askVolPos].cumsum()
bookVolAsk /= bookVolAsk.iloc[-1]

bookVolBid = book.loc[eventIdx,bidVolPos].cumsum()
bookVolBid /= bookVolBid.iloc[-1]

plt.figure()

plt.step(range(1,lvl+1), bookVolAsk, linewidth = 2.5, color="green")
plt.step(range(1,lvl+1), -1*bookVolBid, linewidth = 2.5, color="red")
plt.title("LOB Relative Depth @ %d seconds" %mess.loc[eventIdx,0])
plt.xlabel("Level")
plt.ylabel("% of Volume")

plt.show()

maxLvl = 3
askVolPos = np.arange(1,maxLvl*4,4)

bookVolAsk = book.loc[:,askVolPos]
bookVolBid = book.loc[:,askVolPos+2]


bookVolAsk = bookVolAsk/100.
bookVolBid = bookVolBid/100.

bookVolAsk = bookVolAsk.cumsum(axis=1)
bookVolBid = bookVolBid.cumsum(axis=1)

maxVol = max(max(bookVolAsk.iloc[:,-1]),max(bookVolBid.iloc[:,-1]))
maxVol = np.floor(maxVol*1.1/10)*10


plt.figure()
plt.plot(mess.loc[:,0]/(60*60), bookVolAsk, linewidth=0.5)
plt.plot(mess.loc[:,0]/(60*60), -1*bookVolBid, linewidth=0.5)
plt.ylim(-maxVol,maxVol)
plt.xlim(9.5,16)
plt.show()
