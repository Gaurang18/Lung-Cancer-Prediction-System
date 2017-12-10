import numpy as np
import pandas as pd
from sklearn import cross_validation
import xgboost as xgb
import os
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from sklearn import decomposition
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy.stats import gmean
from sklearn.neighbors import KNeighborsRegressor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.externals import joblib
from sklearn import model_selection
import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler

class OptTrain:
    def __init__(self, trainX, trainY):
        self.trainX = trainX
        self.trainY = trainY

        self.level0 = xgb.XGBClassifier(learning_rate=0.325,
                                       silent=True,
                                       objective="binary:logistic",
                                       nthread=-1,
                                       gamma=0.85,
                                       min_child_weight=5,
                                       max_delta_step=1,
                                       subsample=0.85,
                                       colsample_bytree=0.55,
                                       colsample_bylevel=1,
                                       reg_alpha=0.5,
                                       reg_lambda=1,
                                       scale_pos_weight=1,
                                       base_score=0.5,
                                       seed=0,
                                       missing=None,
                                       n_estimators=1920, max_depth=6)
        self.h_param_grid = {'max_depth': hp.quniform('max_depth', 1, 13, 1),
                        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
                        'learning_rate': hp.quniform('learning_rate', 0.025, 0.5, 0.025),
                        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
                        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
                        'n_estimators': hp.quniform('n_estimators', 100, 1000, 5),
                        }
        self.to_int_params = ['n_estimators', 'max_depth']

    def change_to_int(self, params, indexes):
        for index in indexes:
            params[index] = int(params[index])

    # Hyperopt Implementatation
    def score(self, params):
        self.change_to_int(params, self.to_int_params)
        self.level0.set_params(**params)
        score = model_selection.cross_val_score(self.level0, self.trainX, self.trainY, cv=5, n_jobs=-1)
        print('%s ------ Score Mean:%f, Std:%f' % (params, score.mean(), score.std()))
        return {'loss': score.mean(), 'status': STATUS_OK}

    def optimize(self):
        trials = Trials()
        print('Tuning Parameters')
        best = fmin(self.score, self.h_param_grid, algo=tpe.suggest, trials=trials, max_evals=200)

        print('\n\nBest Scoring Value')
        print(best)

        self.change_to_int(best, self.to_int_params)
        self.level0.set_params(**best)
        self.level0.fit(self.trainX, self.trainY)
        joblib.dump(self.level0,'model_best.pkl', compress=True)

def show_values(pc, fmt="%.2f", **kw):

    from itertools import izip
    pc.update_scalarmappable()
    ax = pc.get_axes()
    for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    
    fig, ax = plt.subplots()
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      
    plt.xlim( (0, AUC.shape[1]) )
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    plt.colorbar(c)
    show_values(c)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    fig = plt.gcf()
    fig.set_size_inches(cm2inch(figure_width, figure_height))


def plot_classification_report(classification_report, title='Classification Report ', cmap='RdBu'):
    
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        print(v)
        plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)



def train_xgboost():
    for parameter in xrange(1,32,5):
        df = pd.read_csv('/home/kshitij/Desktop/Dataset/stage1_labels.csv')
        x = []
        y = []
        did = df['id'].tolist()
        cancer = df['cancer'].tolist()
        for i in range(len(df)):
            if(cancer[i] == 1):
                if os.path.isfile('/home/kshitij/Desktop/Dataset/stage1_features/%s.npy' % did[i]):
                    f = np.load('/home/kshitij/Desktop/Dataset/stage1_features/%s.npy' % did[i])
                    f = f.reshape(f.shape[0], 2048)
                    f = (np.mean(f, axis=0))

                if os.path.isfile('/home/kshitij/Desktop/Dataset/stage1/%s.npy' % did[i]):
                    g = np.load('/home/kshitij/Desktop/Dataset/stage1/%s.npy' % did[i])
                    g = g.flatten()
                    l = g[0:512]
                    g = np.concatenate((g, l), axis=0)
                    l = g[0:1024]
                    g = np.concatenate((l, l), axis=0)
                feats = np.concatenate((f, g),axis =0)

                x.append(feats)
                y.append(cancer[i])
        x = np.array(x)
        y = np.array(y)
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y, random_state=82, test_size=0.1)
        clf = KNeighborsRegressor(parameter)
        clf.fit(X_train, y_train)
        
    return clf

def make_submission():

    x = []
    
    clf = train_xgboost()
    df = pd.read_csv('/home/kshitij/Desktop/Dataset/stage2_sample_submission.csv')
    did = df['id'].tolist()
    for i in range(len(df)):
        if os.path.isfile('/home/kshitij/Desktop/Dataset/stage2_features/%s.npy' % did[i]):
            f = np.load('/home/kshitij/Desktop/Dataset/stage2_features/%s.npy' % did[i])
            f.resize(f.shape[0], 2048)
            f = (np.mean(f, axis=0))

        if os.path.isfile('/home/kshitij/Desktop/Dataset/stage2/%s.npy' % did[i]):
            g = np.load('/home/kshitij/Desktop/Dataset/stage2/%s.npy' % did[i])
            g = g.flatten()
            l = g[0:512]
            g = np.concatenate((g, l), axis=0)
            l = g[0:1024]
            g = np.concatenate((l, l), axis=0)
        feats = np.concatenate((f, g),axis =0)
        
        x.append(feats)

    x = np.array(x)
    score = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    print score
    cm = confusion_matrix(y_test, y_pred)
    plt.matshow(cm)
    plt.title('Confusion matrix ' + str(parameter))
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    p = 'Confusion matrix ' + str(parameter)
    plt.savefig(p+'.png')
    plt.show(block = False)
    plot_classification_report(classification_report(y_test, y_pred))
    p = 'classification_report ' + str(parameter)
    plt.savefig(p+'.png', dpi=200, format='png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    make_submission()

