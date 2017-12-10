
import numpy as np
import pandas as pd
from sklearn import cross_validation
import xgboost as xgb
import os
from sklearn.metrics import classification_report
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
from sklearn.ensemble import VotingClassifier
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from sklearn.metrics import confusion_matrix
#import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_log_error,r2_score,log_loss,roc_curve

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



def plot_classification_report(classification_report, title='Cofusion matrix ', cmap='RdBu'):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857 
    '''
    lines = classification_report

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

h = .02  # step size in the mesh
fig = tools.make_subplots(rows=11, cols=3,
                         print_grid=False)

#h = .02  # step size in the mesh

def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []
    
    for k in range(pl_entries):
        C = map(np.uint8, np.array(cmap(k*h)[:3])*255)
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
        
    return pl_colorscale


names = ["Nearest Neighbors", "Linear SVM", 
         "RBF SVM", "Gaussian Process","Decision Tree", 
         "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA","SGDClassifier"]

classifiers = [
    KNeighborsClassifier(20),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=.2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=8,presort = True),
    RandomForestClassifier(max_depth=8, n_estimators=100, max_features=10),
    MLPClassifier(alpha=3),
    AdaBoostClassifier(learning_rate=.05,n_estimators=100,algorithm="SAMME"),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    SGDClassifier(loss="hinge", alpha=0.01, max_iter=200, fit_intercept=True)
    ]


def train_xgboost():
    df = pd.read_csv('/home/kshitij/Desktop/Dataset/stage1_labels.csv')

    x = []
    y = []
    i = 1
    j = 1
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
    Acc = []
    clfs = []
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=422)
    target_names =["No Cancer","Cancer"]
    for name, clf in zip(names, classifiers):
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            clfs.append(clf)
            y_pred = clf.predict(X_test)
            score = log_loss(y_test,y_pred)
            print score
            Acc.append(score)
        print score

    trace0 = go.Scatter(
    x = names,
    y = Acc,
    name = 'Acc',
    line = dict(color = ('rgb(82, 99, 27)'),
        width = 7)
    )

    data = [trace0]

    # Edit the layout
    layout = dict(title = 'Loss vs Model Plot (Resnet)',
                  xaxis = dict(title = 'Regression'),
                  yaxis = dict(title = 'Log Loss'),
                  )

    fig = dict(data=data, layout=layout)
    py.plot(fig, filename='Loss curve Resnet')

def make_submission():
    clfs = train_xgboost()
    df = pd.read_csv('/home/kshitij/Desktop/Dataset/stage2_sample_submission.csv')
    x = np.array([np.mean(np.load('/home/kshitij/Desktop/Dataset/stage2_features/%s.npy' % str(did)).reshape(-1, 2048), axis=0)
                  for did in df['id'].tolist()])
    preds = []

    for clf in clfs:
        preds.append(np.clip(clf.predict(x), 0.001, 1))

    pred = gmean(np.array(preds), axis=0)

    df['cancer'] = pred
    df.to_csv('submGBR2.csv', index=False)


if __name__ == '__main__':
    make_submission()
    train_xgboost()

