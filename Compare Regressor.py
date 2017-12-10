
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
from sklearn.neighbors import KNeighborsClassifier
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
from sklearn.ensemble import VotingClassifier,GradientBoostingRegressor,RandomTreesEmbedding,IsolationForest,ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_log_error,r2_score,log_loss,roc_auc_score,precision_recall_curve,roc_curve
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn.metrics import precision_recall_fscore_support,average_precision_score,recall_score
from sklearn.metrics import classification_report
from plotly import tools
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
#from sklearn.metrics import mean_squared_log_error
h = .02  # step size in the mesh
#class_weights = {0:1.178,1:6.6}
def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857 
    By HYRY
    '''
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
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857 
    - https://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()    
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    # resize 
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))



def plot_classification_report(classification_report, title='Classification Report ', cmap='RdBu'):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857 
    '''
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

names = ["KNeighborsRegressor","Bagging KNN Essembler",
"XGBoost","GradientBoostingRegressor","AdaBoost + KNN","AdaBoost", "Random Forest",
"SVC + RandomForest Pipeline",
"ExtraTreesRegressor"]

classifiers = [
    KNeighborsRegressor(13),
    KNeighborsRegressor(26),
    BaggingRegressor(KNeighborsRegressor(n_neighbors=30)),
    xgb.XGBRegressor(max_depth=6,
                                       n_estimators=55, #55
                                       learning_rate=0.05,
                                       min_child_weight=60,
                                       nthread=8,
                                       subsample=0.95, #95
                                       colsample_bytree=0.95, # 95
                                       # subsample=1.00,
                                       # colsample_bytree=1.00,
                                       seed=482),
    xgb.XGBRegressor(n_estimators=300, max_depth=7, min_child_weight=2,
                               learning_rate=0.01, subsample=0.80, colsample_bytree=0.70,
                               seed=818, reg_alpha=0.1),

        
    GradientBoostingRegressor(n_estimators=250, max_depth=7,
                                learning_rate=.1, min_samples_leaf=2,
                                min_samples_split=2),

    AdaBoostRegressor(KNeighborsRegressor(26)),
    AdaBoostRegressor(),
    #DecisionTreeRegressor(max_depth=7),
    RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=4, n_jobs=-1, random_state=882),
    Pipeline([
                ('feature_selection', SelectFromModel(LinearSVC(penalty="l2"))),
                ('Regression',RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=4, max_features=0.4, n_jobs=-1, random_state=0))
            ]),
    ExtraTreesRegressor(n_estimators=10, criterion='mse',max_depth=8, min_samples_split=4, min_samples_leaf=2,warm_start=False),
    ]

def train_xgboost():
    df = pd.read_csv('/home/kshitij/Desktop/Dataset/stage1_labels.csv')


    x = []
    y = []
    did = df['id'].tolist()
    cancer = df['cancer'].tolist()
    for i in range(len(df)):
        g = []
        if os.path.isfile('/home/kshitij/Desktop/Dataset/stage1/%s.npy' % did[i]):
            f = np.load('/home/kshitij/Desktop/Dataset/stage1/%s.npy' % did[i])
            f = f.reshape(-1,324*13)
            f = f[:,0:4096]
            f = f.reshape(-1,2048)
            #print f.shape
            g1 = np.mean(f, axis=0)
            
        if os.path.isfile('/home/kshitij/Desktop/Dataset/stage1_features/%s.npy' % did[i]):
            f = np.load('/home/kshitij/Desktop/Dataset/stage1_features/%s.npy' % did[i])
            f = f.reshape(-1,2048)
            g2 = np.mean(f, axis=0)
            y.append(cancer[i])

        lp = g1+g2
        x.append(lp)
            
    x = np.array(x)
    y = np.array(y)
    Acc = []
    logerr = []
    X_train, X_test, y_train, y_true = train_test_split(x, y, test_size=.3, random_state=425)
    # oversampler=SMOTE(random_state=474)
    # #X_train.reshape()
    # #y_train.reshape()
    clfs = []
    #X_train,y_train=oversampler.fit_sample(X_train,y_train)
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        clfs.append(clf)
        y_pred = clf.predict(X_test)
        score = average_precision_score(y_true, y_pred,average = 'micro')
        print score
        score = roc_auc_score(y_true, y_pred,average = 'micro')
        print score
        Acc.append(score)
        print score


if __name__ == '__main__':
    #make_submission()
    train_xgboost()

