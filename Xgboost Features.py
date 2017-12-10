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

def train_xgboost():
    df = pd.read_csv('/home/kshitij/Desktop/Dataset/stage1_labels.csv')
    did = df['id'].tolist()
    x = get_trn()
    y= df['cancer']
    skf = StratifiedKFold(n_splits=25, random_state=88, shuffle=True)

    clfs = []
    for train_index, test_index in skf.split(x, y):
        trn_x, val_x = x[train_index,:], x[test_index,:]
        trn_y, val_y = y[train_index], y[test_index]

        clf = xgb.XGBRegressor(max_depth=10,
                               n_estimators=2000,
                               min_child_weight=9,
                               learning_rate=0.04,
                               nthread=8,
                               subsample=0.80,
                               colsample_bytree=0.80,
                               seed=8888)
        clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=2)
        clfs.append(clf)

    return clfs

def make_submite():
    clfs = train_xgboost()

    df = pd.read_csv('/home/kshitij/Desktop/Dataset/stage2_sample_submission.csv')

    x = get_tst()

    preds = []
    for clf in clfs:
        preds.append(np.clip(clf.predict(x),0.001,1))

    pred = gmean(np.array(preds), axis=0)
    print pred
    for i in range(len(pred)):
       df['cancer'] = pred
    df.to_csv('subm_xgb.csv', index=False)
    print(df.head())
    
def make_ensemble():
    df1 = pd.read_csv('subm_xgb.csv')
    df2 = pd.read_csv('subm_krs.csv')
    df1['cancer'] = df1['cancer'] * 0.8 + df2['cancer'] * 0.2
    df1.to_csv('ensemble.csv', index=False)

if __name__ == '__main__':

    make_submite()
