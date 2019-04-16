'''

John Heisey, M.Sc. Computational Science & Engineering
February 11, 2019

Code purpose:
This script truncates the continous limit order book data
to only exist within standard trading hours using NOII data
'''

import pandas as pd
import numpy as np
import csv
import os
import sys


nameFile = sys.argv[1]
tickers = ["AAPL", "FB", "GOOG", "INTC", "MSFT"]

path = nameFile+"/mids"
os.mkdir(path)

for ticker in tickers:

	nameBook = nameFile+'/books/books_'+ticker+'.txt'
	nameMess = nameFile+'/messages/messages_'+ticker+'.txt'
	nameNOII = nameFile+'/noii/noii_'+ticker+'.txt'
	nameTrades = nameFile+'/trades/trades_'+ticker+'.txt'

	book = pd.read_csv(nameBook, sep=',', header=1)
	mess = pd.read_csv(nameMess, sep=',', header=1)
	noii = pd.read_csv(nameNOII, sep=',', header=1)
	trades = pd.read_csv(nameTrades, sep=',', header=1)

	crosses = noii[noii.iloc[:,3] == 'Q']

	book = book[(book.iloc[:,0] >= crosses.iloc[0,0]) & (book.iloc[:,1] > crosses.iloc[0,1])]
	book = book[(book.iloc[:,0] <= crosses.iloc[1,0]) & (book.iloc[:,1] > crosses.iloc[1,1])]

	mess = mess[(mess.iloc[:,0] >= crosses.iloc[0,0]) & (mess.iloc[:,1] > crosses.iloc[0,1])]
	mess = mess[(mess.iloc[:,0] <= crosses.iloc[1,0]) & (mess.iloc[:,1] > crosses.iloc[1,1])]

	trades = trades[(trades.iloc[:,0] >= crosses.iloc[0,0]) & (trades.iloc[:,1] > crosses.iloc[0,1])]
	trades = trades[(trades.iloc[:,0] <= crosses.iloc[1,0]) & (trades.iloc[:,1] > crosses.iloc[1,1])]

	if ticker == 'AAPL':
		midprice = (book.iloc[:,3] + book.iloc[:,13])/14
	else:
		midprice = (book.iloc[:,3] + book.iloc[:,13])/2

	times = book.ix[:,[0,1]]

	# print times

	csvfile = nameFile+"_mid"


	
	midprice = [times, midprice]
	midprice = pd.concat(midprice, axis=1)
	
	midprice.to_csv(nameFile+'/mids/mid_'+ticker+'.csv', index=False, header=False)
	book.to_csv(nameFile+'/books/books_'+ticker+'.txt', index=False, header=False)
	mess.to_csv(nameFile+'/messages/messages_'+ticker+'.txt', index=False, header=False)
	trades.to_csv(nameFile+'/trades/trades_'+ticker+'.txt', index=False, header=False)
