import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from scipy.stats import gmean
import keras as k
import keras.layers as l
from keras.layers import Dense, Activation
import os 
import tensorflow as tf
from keras.layers import Merge, Input
from keras.models import Sequential

def get_trn():
    df = pd.read_csv('/home/kshitij/Desktop/Dataset/stage1_labels.csv')
    did = df['id'].tolist()
    x = []
    for i in range(len(df)):
        if os.path.isfile('/home/kshitij/Desktop/Dataset/stage1_features/%s.npy' % did[i]):
            f = np.load('/home/kshitij/Desktop/Dataset/stage1_features/%s.npy' % did[i])
            f = f.reshape(f.shape[0], 2048)
            f = (np.mean(f, axis=0))
            #print f
            x.append(f)

    trn_x = np.array(x)
    
    return trn_x

def get_tst():
    df_test = pd.read_csv('/home/kshitij/Desktop/Dataset/stage2_sample_submission.csv')
    did = df_test['id'].tolist()
    x = []
    for i in range(len(df_test)):
        if os.path.isfile('/home/kshitij/Desktop/Dataset/stage2_features/%s.npy' % did[i]):
            f = np.load('/home/kshitij/Desktop/Dataset/stage2_features/%s.npy' % did[i])
            f = f.reshape(-1, 2048)
            f = (np.mean(f, axis=0))
            x.append(f)

    trn_x = np.array(x)
    return trn_x

def get_model(size):

    m = Sequential([
        Dense(128, input_shape=(size,)),
        Activation('relu'),
        Dense(32),
        Activation('relu'),
        Dense(1),
        Activation('sigmoid')
    ])
    m.compile(loss='binary_crossentropy', optimizer='adam')
    return m
    
def keras():
    
    df = pd.read_csv('/home/kshitij/Desktop/Dataset/stage1_labels.csv')
    did = df['id'].tolist()
    y = df['cancer'].as_matrix()
    x = get_trn()
    x_test = get_tst()
    skf = StratifiedKFold(n_splits=25, random_state=88, shuffle=True)
    df_test = pd.read_csv('/home/kshitij/Desktop/Dataset/stage2_sample_submission.csv')
    preds = []
    for train_index, test_index in skf.split(x, y):
        trn_x, val_x = x[train_index,:], x[test_index,:]
        trn_y, val_y = y[train_index], y[test_index]

        m = get_model(trn_x.shape[1])
        #print "Hello"
        #print trn_x.shape
        #print trn_y.shape
        #trn_x = trn_x
        #trn_x = trn_x.reshape((1, -1))

        #val_x = val_x.reshape((-1, 1340))
        # m.compile(optimizer=Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
        #                   loss='binary_crossentropy', metrics=['accuracy']), validation_data=(val_x, val_y),
        m.fit(trn_x, trn_y, batch_size=32, nb_epoch=20,verbose=1)
        #print "Hello"
        pred = [p[0] for p in m.predict(x_test)]
        preds.append(pred)
        
    preds = np.array(preds)
    print(preds)
    print(preds.shape)
    
    preds = preds.mean(axis=0)
    df_test['cancer'] = preds
    df_test.to_csv('subm_krs.csv', index=False)


if __name__ == '__main__':
    keras()