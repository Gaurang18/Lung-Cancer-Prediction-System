
import numpy as np
import pandas as pd
from sklearn import cross_validation
import xgboost as xgb
import os
from scipy.stats import gmean
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.semi_supervised import LabelSpreading
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import VotingClassifier,GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import log_loss

def importance_XGB(clf):
    impdf = []
    for ft, score in clf.booster().get_fscore().iteritems():
        impdf.append({'feature': ft, 'importance': score})
    impdf = pd.DataFrame(impdf)
    impdf = impdf.sort_values(by='importance', ascending=False).reset_index(drop=True)
    impdf['importance'] /= impdf['importance'].sum()
    return impdf

def train_xgboost():
    df = pd.read_csv('/home/kshitij/Desktop/Dataset/stage1_labels.csv')
    x = []
    y = []
    did = df['id'].tolist()
    cancer = df['cancer'].tolist()
    for i in range(len(df)):
        if os.path.isfile('/home/kshitij/Desktop/Dataset/stage1 hog/%s.npy' % did[i]):
            f = np.load('/home/kshitij/Desktop/Dataset/stage1 hog/%s.npy' % did[i])
            f = f.reshape(-1, 324*13)
            x.append(np.mean(f, axis=0))
            print f.shape
            y.append(cancer[i])

    x = np.array(x)
    y = np.array(y)

    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=822, stratify=y, test_size=0.1)

    clfs = []
    for s in range(4):
        clf = xgb.XGBRegressor(n_estimators=1000, max_depth=10, min_child_weight=10,
                               learning_rate=0.01, subsample=0.80, colsample_bytree=0.70,
                               seed=822 + s, reg_alpha=0.1)

        clf = BaggingRegressor(KNeighborsRegressor(n_neighbors=30))
        
        clf.fit(x,y)
        clfs.append(clf)
        
    return clfs


def make_submission():
    clfs = train_xgboost()
    x = []
    df = pd.read_csv('/home/kshitij/Desktop/Dataset/stage2_sample_submission.csv')
    did = df['id'].tolist()
    for i in range(len(df)):
        f = np.load('/home/kshitij/Desktop/Dataset/stage2 hog/%s.npy' % did[i])
        f = f.reshape(-1, 324*13)
        x.append(np.mean(f, axis=0))
    #print f.shape
    preds = []
    x = np.array(x)
    for clf in clfs:
        preds.append(np.clip(clf.predict(x), 0.001, 1))

    pred = gmean(np.array(preds), axis=0)

    y_true = [0,0,1,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,1,1,0,0,1,0,1,
    1,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,
    1,1,0,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,
    0,0,0,0,0,0,1,0,1,1,0,0,1,0,0,1,0,1,1,0,0,1,0,1,1,0,0,1,0,1,1,1,1,0,0,0,1,0,1,0,0,1,1,0,1,0,0,0,0,0,1,1,0,1,0,1,0,1,0,
    0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,
    0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,1,1,1,0,1,1,0,0,0,0,0,0,1,0,0,1,1,0,0,0,1,0,0,1,0,1,0,1,1,1,
    0,0,0,0,0,1,0,0,0,1,0,0,1,1,1,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,1,0,1,1,1,0,0,1,0,0,1,1,0,0,0,0,0,1,0,1,1,0,0,0,0,
    0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,1,1,1,0,0,0,1,0,0,
    1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,1,0,1,0,1,0]

    df['cancer'] = pred
    df.to_csv('hogfsubm.csv', index=False)
    pred = np.array(pred)
    y_true = np.array(y_true)
    print np.mean(log_loss(pred,y_true))
    ap = np.concatenate(log_loss(pred,y_true)).astype(None)
    print ap


if __name__ == '__main__':
    make_submission()

