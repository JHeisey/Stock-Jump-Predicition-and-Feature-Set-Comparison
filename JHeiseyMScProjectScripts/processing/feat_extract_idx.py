'''

John Heisey, M.Sc. Computational Science & Engineering
February 11, 2019

Code purpose:
This script extracts feature vectors from NASDAQ limit order book
and message data for the best 10 bids/asks. It does so by extracting
processed information at prespecified indices from 'idx' csv file.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from scipy.stats import moment,skew,kurtosis
import sys

ticker = sys.argv[1]

nameBook = ticker+"_books_full.csv"
nameMess = ticker+"_messages_full.csv"
nameIdx = ticker+"_idx_1s_year.csv"
nameSent = ticker+"_sent_prev.csv"
book = pd.read_csv(nameBook, sep=',', header=0)
msg = pd.read_csv(nameMess, sep=',', header=0)
idx = pd.read_csv(nameIdx, sep=',', header=0)
sent = pd.read_csv(nameSent, sep=',', header=0)

msg = msg.iloc[:,1:]
msg.columns = ['sec','nano','name','type','refno','side','shares','price','mpid']

book = book.iloc[:,1:]

# below useful for debugging purposes (truncates data for readability)
#np.set_printoptions(precision=2, suppress=True)



def feat_extract_kz(t,t_1s,t_2s,T, T_1s,book,msg,day):
        v = book.iloc[t,3:].values
        sec = book.iloc[t,0]
        v_1t = book.iloc[t_1s,3:].values
        msg_t = msg.iloc[t_1s:t+1,[3,5]]
        msg_T = msg.iloc[T_1s:T+1,[3,5]]
        msg_t1s = msg.iloc[t_2s:t_1s+1,[3,5]]

        lam_T = np.array([msg_T[((msg_T.type == 'A') | (msg_T.type == 'F'))&(msg_T.side == 'S')].count().values[0],\
                           msg_T[((msg_T.type == 'A') | (msg_T.type == 'F'))&(msg_T.side == 'B')].count().values[0],\
                           msg_T[((msg_T.type == 'E') | (msg_T.type == 'C') | (msg_T.type == 'P'))&(msg_T.side == 'S')].count().values[0],\
                           msg_T[((msg_T.type == 'E') | (msg_T.type == 'C') | (msg_T.type == 'P'))&(msg_T.side == 'B')].count().values[0],\
                           msg_T[((msg_T.type == 'X') | (msg_T.type == 'D'))&(msg_T.side == 'S')].count().values[0],\
                           msg_T[((msg_T.type == 'X') | (msg_T.type == 'D'))&(msg_T.side == 'B')].count().values[0],\
                           msg_T[msg_T.type.isin(['A','F','E','C','E','X','D','U','P'])].count().values[0]])

        lam_t = np.array([msg_t[((msg_t.type == 'A') | (msg_t.type == 'F'))&(msg_t.side == 'S')].count().values[0],\
                           msg_t[((msg_t.type == 'A') | (msg_t.type == 'F'))&(msg_t.side == 'B')].count().values[0],\
                           msg_t[((msg_t.type == 'E') | (msg_t.type == 'C') | (msg_t.type == 'P'))&(msg_t.side == 'S')].count().values[0],\
                           msg_t[((msg_t.type == 'E') | (msg_t.type == 'C') | (msg_t.type == 'P'))&(msg_t.side == 'B')].count().values[0],\
                           msg_t[((msg_t.type == 'X') | (msg_t.type == 'D'))&(msg_t.side == 'S')].count().values[0],\
                           msg_t[((msg_t.type == 'X') | (msg_t.type == 'D'))&(msg_t.side == 'B')].count().values[0],\
                           msg_t[msg_t.type.isin(['A','F','E','C','E','X','D','U','P'])].count().values[0]])

        lam_t1s = np.array([msg_t1s[((msg_t1s.type == 'A') | (msg_t1s.type == 'F'))&(msg_t1s.side == 'S')].count().values[0],\
                           msg_t1s[((msg_t1s.type == 'A') | (msg_t1s.type == 'F'))&(msg_t1s.side == 'B')].count().values[0],\
                           msg_t1s[((msg_t1s.type == 'E') | (msg_t1s.type == 'C') | (msg_t1s.type == 'P'))&(msg_t1s.side == 'S')].count().values[0],\
                           msg_t1s[((msg_t1s.type == 'E') | (msg_t1s.type == 'C') | (msg_t1s.type == 'P'))&(msg_t1s.side == 'B')].count().values[0],\
                           msg_t1s[((msg_t1s.type == 'X') | (msg_t1s.type == 'D'))&(msg_t1s.side == 'S')].count().values[0],\
                           msg_t1s[((msg_t1s.type == 'X') | (msg_t1s.type == 'D'))&(msg_t1s.side == 'B')].count().values[0],\
                           msg_t1s[msg_t1s.type.isin(['A','F','E','C','E','X','D','U','P'])].count().values[0]])



        v1 = v
        v2 = np.concatenate((v[10:20]-v[0:10],(v[0:10]+v[10:20])/2), axis=None)
        v3 = np.concatenate((v[19]-v[10],v[0]-v[9], abs(v[11:20]-v[10:19]), abs(v[1:10]-v[0:9])),axis=None)
        v4 = np.concatenate((np.sum(v[10:20])/10,np.sum(v[0:10])/10,np.sum(v[30:40])/10,np.sum(v[20:30])/10),axis=None)
        v5 = np.concatenate((np.sum(v[10:20]-v[0:10]),np.sum(v[30:40]-v[20:30])),axis=None)

        v6 = np.concatenate((v[10:20]-v_1t[10:20],v[0:10]-v_1t[0:10],v[30:40]-v_1t[30:40],v[20:30]-v_1t[20:30]),axis=None)
        v7 = lam_t
        v8 = np.array([1 if num > 0 else 0 for num in(lam_t-lam_T)])
        v9 = np.concatenate((lam_t[0]-lam_t1s[0],\
                                             lam_t[1]-lam_t1s[1],\
                                             lam_t[2]-lam_t1s[2],\
                                             lam_t[3]-lam_t1s[3],\
                                             lam_t[4]-lam_t1s[4],\
                                             lam_t[5]-lam_t1s[5],\
                                             lam_t[6]-lam_t1s[6]),axis=None)
        v10 = np.round((book.iloc[t,0])/3600.0)

        V = np.concatenate((v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,sec,day),axis=None)
        return V


def feat_extract_jh(t,book,sent,day):
        # total feature number: 102 
        v = book.iloc[t,3:].values
        bidSum = np.repeat(v[0:10].tolist(),v[20:30].tolist())
        askSum = np.repeat(v[10:20].tolist(),v[30:40].tolist())
        sec = book.iloc[t,0]
        v1 = v
        v2 = np.concatenate((v[10:20]-v[0:10],(v[0:10]+v[10:20])/2), axis=None)
        v3 = np.concatenate((v[19]-v[10],v[0]-v[9], abs(v[11:20]-v[10:19]), abs(v[1:10]-v[0:9])),axis=None)
        v4 = np.concatenate((np.sum(v[10:20])/10,np.sum(v[0:10])/10,np.sum(v[30:40])/10,np.sum(v[20:30])/10),axis=None)
        v5 = np.concatenate((np.sum(v[10:20]-v[0:10]),np.sum(v[30:40]-v[20:30])),axis=None)
        v6 = np.array([np.mean(askSum),np.std(askSum), skew(askSum), kurtosis(askSum), np.mean(bidSum),np.std(bidSum), skew(bidSum), kurtosis(bidSum)])
        v7 = np.round((book.iloc[t,0])/3600.0)
        v8 = sent.iloc[day-1,3:]

        V = np.concatenate((v1,v2,v3,v4,v5,v6,v7,v8,sec,day),axis=None)
        return V

sml_range = int(len(idx.iloc[:,0])/30)
lrg_range = len(idx.iloc[:,0])

kz_full_feat = np.zeros((lrg_range,150))

full_feat = np.zeros((lrg_range,102))

for i in range(lrg_range):
    if i == 0:
        print("Started running..")

    t = idx.iloc[i,0]
    day = idx.iloc[i,1] 
    full_feat[i,:] = feat_extract_jh(t, book, sent, day)
    kz_full_feat[i,:] = feat_extract_kz(t,t-1,t-2,t-900,t-901, book, msg, day)


feat_csv = ticker+"_features_jh_new.csv"
kz_feat_csv = ticker+"_features_kz.csv"
np.savetxt(feat_csv,full_feat, delimiter=",", fmt='%1.2f')
print("Feat vec (John Heisey) generated for "+ticker+".")

np.savetxt(kz_feat_csv,kz_full_feat, delimiter=",", fmt='%1.2f')
print("Feat vec (Kercheval and Zhang) generated for "+ticker+".")
