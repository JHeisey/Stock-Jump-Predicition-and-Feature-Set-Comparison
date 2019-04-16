'''

John Heisey, M.Sc. Computational Science & Engineering
February 11, 2019

Code purpose:
This script unzips the raw data obtained through NASDAQ
'''

import gzip
import time

tic = time.clock()

inF = gzip.GzipFile("raw_data/S010313-v41.txt.gz", 'rb')
s = inF.read()
inF.close()

outF = file("processed_data/S010313-v41.txt", 'wb')
outF.write(s)
outF.close()

toc = time.clock()

print("Time taken: "+str(toc-tic))
