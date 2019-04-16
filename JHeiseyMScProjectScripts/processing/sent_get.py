'''

John Heisey, M.Sc. Computational Science & Engineering
February 11, 2019

Code purpose:
This script obtains the sentiment data from quandl
'''

import numpy as np
import quandl
import pandas as pd
import datetime
date1 = '2012-01-01'
date2 = '2012-12-31'
start = datetime.datetime.strptime(date1, '%Y-%m-%d')
end = datetime.datetime.strptime(date2, '%Y-%m-%d')

step = datetime.timedelta(days=1)

dates = []

while start <= end:
    dates.append(str(start.date()))
    start += step

print(dates)





quandl.ApiConfig.api_key = "2aYSz2x9fcDHz6oU3NYh"

#dates = pd.date_range(start='1/1/2013', end='12/31/2013')

#dates = [i.replace("'", '') for i in dates]

data = quandl.get_table('IFT/NSA', ticker='AAPL', date=dates)

print(data)
