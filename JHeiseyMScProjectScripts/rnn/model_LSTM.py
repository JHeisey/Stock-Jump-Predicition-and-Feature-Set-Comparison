'''

John Heisey, M.Sc. Computational Science & Engineering
February 11, 2019

Code purpose:
Accepts input from main script and builds and trains a
sequential RNN model in Keras

Output: Returns individual model results for each dataset
'''

from __future__ import print_function
from sklearn import preprocessing
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
import pandas as pd
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.callbacks import ModelCheckpoint

def LSTM_model_train(data, epochs, test_per, name):

    def f1(y_true, y_pred):
        y_pred = K.round(y_pred)
        tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
        fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        f1 = 2*p*r / (p+r+K.epsilon())
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        return K.mean(f1)

    def precision(y_true, y_pred):  
        """Precision metric.    
        Only computes a batch-wise average of precision. Computes the precision, a
        metric for multi-label classification of how many selected items are
        relevant.
        """ 
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))  
        precision = true_positives / (predicted_positives + K.epsilon())    
        return precision

    def recall(y_true, y_pred): 
        """Recall metric.   
        Only computes a batch-wise average of recall. Computes the recall, a metric
        for multi-label classification of how many relevant items are selected. 
        """ 
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))   
        recall = true_positives / (possible_positives + K.epsilon())    
        return recall

    def f1_score(y_true, y_pred):
        """Computes the F1 Score
        Only computes a batch-wise average of recall. Computes the recall, a metric
        for multi-label classification of how many relevant items are selected. 
        """
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        return (2 * p * r) / (p + r + K.epsilon())


    x = data.iloc[:,:-3]
    y = data.iloc[:,-1]
    x = x.fillna(0)


    # LSTM
    lstm_output_size = 40 

    # Training
    batch_size = 60000
    print("Batch size:",batch_size)

    x = x.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    x = pd.DataFrame(x_scaled)

    test_idx = int(x.shape[0]*test_per)

    print("test percentage:", test_per)

    (x_train, y_train) = x.iloc[:test_idx,:].values,y.iloc[:test_idx].values
    (x_test, y_test) = x.iloc[test_idx:,:].values,y.iloc[test_idx:].values


    #reshaping sets to 3D for LSTM training
    (x_train, y_train) = x_train.reshape(-1, np.shape(x_train)[0], np.shape(x_train)[1]), y_train.reshape(-1, np.shape(y_train)[0],1)
    (x_test, y_test) = x_test.reshape(-1, np.shape(x_test)[0], np.shape(x_test)[1]), y_test.reshape(-1, np.shape(y_test)[0],1)


    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')

    seq_length = x_train.shape[1]
    input_dims = x_train.shape[2]

    model = Sequential()

    model.add(LSTM(lstm_output_size, 
                   input_shape=(None, input_dims),
                   return_sequences=True))
    model.add(Dropout(0.35))
    model.add(Dense(40))
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[f1_score, precision, recall])
    
    fileName = 'weights-best-'+name+'.hdf5'
    checkpointer = ModelCheckpoint(filepath=fileName,monitor='val_f1_score',verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)


    print('Train...')

    model.fit(x_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=epochs, validation_data=(x_train,y_train), verbose=1,callbacks=[checkpointer])
    
    model.load_weights(fileName)

    loss, f1, precision, recall = model.evaluate(x_test, y_test, verbose=1)


    print('Test F1 score:', f1)
    print('Test precision:', precision)
    print('Test recall:', recall)
    print('Test loss:', loss)

    return loss, f1, precision, recall



