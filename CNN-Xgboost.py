import numpy as np
import pandas as pd
from sklearn import cross_validation
import xgboost as xgb
import os
from scipy.stats import gmean
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
import plotly.plotly as py
import plotly.graph_objs as go
import plotly

def train_xgboost():
    df = pd.read_csv('/home/kshitij/Desktop/Dataset/stage1_labels.csv')
    x = []
    y = []
    did = df['id'].tolist()
    cancer = df['cancer'].tolist()
    for i in range(len(df)):
        if os.path.isfile('/home/kshitij/Desktop/Dataset/stage1_features/%s.npy' % did[i]):
            f = np.load('/home/kshitij/Desktop/Dataset/stage1_features/%s.npy' % did[i])
            f = f.reshape(f.shape[0], 2048)
            x.append(np.mean(f, axis=0))
            y.append(cancer[i])

    x = np.array(x)
    y = np.array(y)

    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=822, stratify=y, test_size=0.1)
    
    clfs = []
    for s in range(3):
        clf = xgb.XGBRegressor(n_estimators=1200, max_depth=9, min_child_weight=4,
                               learning_rate=0.05, subsample=0.70, colsample_bytree=0.60,
                               seed=822, reg_alpha=0.1)
        a6 = clf.fit(trn_x, trn_y)
        clfs.append(clf)
        features = [str(i) for i in range(20)]
        mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))
        ts = pd.Series(a6.booster().get_fscore())
        data = [go.Bar(
            x =ts,
            y=[p for p in range(len(features))],
            orientation = 'h'
        )]
        py.plot(data, filename='horizontal-bar')
        ts.index = ts.reset_index()['index'].map(mapFeat)
        ts.sort_values()[-100:].plot(kind="barh", title=("features importance"))

    return clfs


def make_submission():
    clfs = train_xgboost()
    df = pd.read_csv('/home/kshitij/Desktop/Dataset/stage2_sample_submission.csv')
    x = np.array([np.mean(np.load('/home/kshitij/Desktop/Dataset/stage2_features/%s.npy' % str(did)).reshape(-1, 2048), axis=0)
                  for did in df['id'].tolist()])
    preds = []

    for clf in clfs:
        preds.append(np.clip(clf.predict(x), 0.0001, 1))
    
    pred = gmean(np.array(preds), axis=0)

    df['cancer'] = pred
    df.to_csv('submfnal2.csv', index=False)

if __name__ == '__main__':
    make_submission()

