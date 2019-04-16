'''

John Heisey, M.Sc. Computational Science & Engineering
February 11, 2019

Code purpose:
This accepts arguments towards one of the given datasets and trains/tests
a recurrent neural network using Keras.

Output: A summary result of the model metrics
'''

import pandas as pd
import numpy as np
from model_LSTM import LSTM_model_train
import sys

ticker = sys.argv[1]
feat = sys.argv[2]
time = sys.argv[3]

if sys.argv[3] == "5min":
        dataName = ticker+"_norm_feat_"+time+"_"+feat+".csv"
elif sys.argv[3] == "30s":
        dataName = ticker+"_norm_feat_"+feat+".csv"
else:
        print("Incorrect arguments (ticker,feat,time)")

print("Working on data:",dataName)
print('Loading data...')

data = pd.read_csv(dataName, header=None)

epochs = 800 #200 for 5min

#constant percentage of full test data
test_per = 0.15 

#convert percentage to array data amount
test_size = int(data.shape[0]*test_per)

#number of nested CV validations
k_amt = 4

#nested CV validation dataset percentages
nest_per = np.linspace((1.-test_per)/k_amt,(1.-test_per),k_amt)

k_losses = []
k_f1s = []
k_precisions = []
k_recalls = []

for i in range(len(nest_per)):
        k_end = int(data.shape[0]*(nest_per[i]+test_per))+1
        k_data = data.iloc[:k_end,:]
        k_test_per = test_per/(test_per + nest_per[i])
        k_name = ticker+"_"+time+"_"+feat+"_CV_"+str(i+1)
        loss, f1, precision, recall = LSTM_model_train(k_data,epochs,k_test_per, k_name)
        k_losses.append(loss)
        k_f1s.append(f1)
        k_precisions.append(precision)
        k_recalls.append(recall)


csv_data = []

print("Nested CV losses:", k_losses)
print("Nested CV f1s:", k_f1s)
print("Nested CV precisions:", k_precisions)
print("Nested CV recalls:", k_recalls)
print("")
print("Avg. loss:", np.mean(k_losses))
print("Avg. f1:", np.mean(k_f1s))
print("Avg. precision:", np.mean(k_precisions))
print("Avg. recall:", np.mean(k_recalls))
print("")
print("SD loss:", np.std(k_losses))
print("SD f1:", np.std(k_f1s))
print("SD precision:", np.std(k_precisions))
print("SD recall:", np.std(k_recalls))

avg_loss = ["Avg. loss:", np.mean(k_losses)]
csv_data.append(avg_loss)
avg_f1 = ["Avg. f1:", np.mean(k_f1s)]
csv_data.append(avg_f1)
avg_prec = ["Avg. precision:", np.mean(k_precisions)]
csv_data.append(avg_prec)
avg_rec = ["Avg. recall:", np.mean(k_recalls)]
csv_data.append(avg_rec)

sd_loss = ["SD loss:", np.std(k_losses)]
csv_data.append(sd_loss)
sd_f1 = ["SD f1:", np.std(k_f1s)]
csv_data.append(sd_f1)
sd_prec = ["SD precision:", np.std(k_precisions)]
csv_data.append(sd_prec)
sd_rec = ["SD recall:", np.std(k_recalls)]
csv_data.append(sd_rec)

k_losses.insert(0,"Nested CV losses:")
csv_data.append(k_losses)
k_f1s.insert(0,"Nested CV f1s:")
csv_data.append(k_f1s)
k_precisions.insert(0,"Nested CV precisions:")
csv_data.append(k_precisions)
k_recalls.insert(0,"Nested CV recalls:")
csv_data.append(k_recalls)

result_csv_name = ticker+"_"+feat+"_"+time+"_NO_MORNING_results_"+str(epochs)+"_epochs.csv"

import csv

with open(result_csv_name, "w") as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)



