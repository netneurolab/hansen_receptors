# -*- coding: utf-8 -*-
"""
Disease analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netneurotools import datasets, stats, utils
from scipy.stats import zscore, pearsonr
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import squareform, pdist
from sklearn.linear_model import LinearRegression

def get_reg_r_sq(X, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    yhat = lin_reg.predict(X)
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    adjusted_r_squared = 1 - (1 - r_squared) * \
        (len(y) - 1) / (len(y) - X.shape[1] - 1)
    return adjusted_r_squared


def cv_slr_distance_dependent(X, y, coords, train_pct=.75, metric='rsq'):
    '''
    cross validates linear regression model using distance-dependent method.
    X = n x p matrix of input variables
    y = n x 1 matrix of output variable
    coords = n x 3 coordinates of each observation
    train_pct (between 0 and 1), percent of observations in training set
    metric = {'rsq', 'corr'}
    '''

    P = squareform(pdist(coords, metric="euclidean"))
    train_metric = []
    test_metric = []

    for i in range(len(y)):
        distances = P[i, :]  # for every node
        idx = np.argsort(distances)

        train_idx = idx[:int(np.floor(train_pct * len(coords)))]
        test_idx = idx[int(np.floor(train_pct * len(coords))):]

        mdl = LinearRegression()
        mdl.fit(X[train_idx, :], y[train_idx])
        if metric == 'rsq':
            # get r^2 of train set
            train_metric.append(get_reg_r_sq(X[train_idx, :], y[train_idx]))

        elif metric == 'corr':
            rho, _ = pearsonr(mdl.predict(X[train_idx, :]), y[train_idx])
            train_metric.append(rho)

        yhat = mdl.predict(X[test_idx, :])
        if metric == 'rsq':
            # get r^2 of test set
            SS_Residual = sum((y[test_idx] - yhat) ** 2)
            SS_Total = sum((y[test_idx] - np.mean(y[test_idx])) ** 2)
            r_squared = 1 - (float(SS_Residual)) / SS_Total
            adjusted_r_squared = 1-(1-r_squared)*((len(y[test_idx]) - 1) /
                                                  (len(y[test_idx]) -
                                                   X.shape[1]-1))
            test_metric.append(adjusted_r_squared)

        elif metric == 'corr':
            rho, _ = pearsonr(yhat, y[test_idx])
            test_metric.append(rho)

    return train_metric, test_metric


"""
set-up
"""

scale = 'scale033'
path = 'C:/Users/justi/OneDrive - McGill University/MisicLab/proj_receptors/\
github/hansen_receptors/'

# set up parcellation
cammoun = datasets.fetch_cammoun2012()
info = pd.read_csv(cammoun['info'])
cortex = info.query('scale == "scale033" & structure == "cortex"')['id']
cortex = np.array(cortex) - 1  # python indexing
coords = utils.get_centroids(cammoun[scale], image_space=True)
coords = coords[cortex, :]
nnodes = len(cortex)

# load the receptor data
receptor_data = np.genfromtxt(path+'results/receptor_data_'+scale+'.csv', delimiter=',')
receptor_names = np.load(path+'data/receptor_names_pet.npy')

# load the enigma cortical thickness cohens d ("atrophy") maps
ct = np.genfromtxt(path+'data/enigma_atrophy.csv', delimiter=',')
disorders = ['22q', 'adhd', 'asd', 'epilepsy_gge', 'epilepsy_rtle',
             'epilepsy_ltle', 'depression', 'ocd', 'schizophrenia',
             'bipolar', 'obesity', 'schizotypy', 'park']

# colourmaps
cmap = np.genfromtxt(path+'data/colourmap.csv', delimiter=',')
cmap_div = ListedColormap(cmap)
cmap_seq = ListedColormap(cmap[128:, :])

"""
Dominance analysis
"""

model_metrics = dict([])
train_metric = np.zeros([nnodes, len(disorders)])
test_metric = np.zeros(train_metric.shape)

for i in range(len(disorders)):
    print(i)
    m, _ = stats.get_dominance_stats(zscore(receptor_data),
                                     zscore(ct[:, i]))
    model_metrics[disorders[i]] = m
    # cross validate the model
    train_metric[:, i], test_metric[:, i] = \
        cv_slr_distance_dependent(zscore(receptor_data), zscore(ct[:, i]),
                                  coords, .75, metric='corr')

dominance = np.zeros((len(disorders), len(receptor_names)))

for i in range(len(model_metrics)):
    tmp = model_metrics[disorders[i]]
    dominance[i, :] = tmp["total_dominance"]
np.save(path+'results/dominance_enigma.npy', dominance)
np.save(path+'results/enigma_cv_train.npy', train_metric)
np.save(path+'results/enigma_cv_test.npy', test_metric)

plt.ion()
plt.figure()
sns.heatmap(dominance, xticklabels=receptor_names, yticklabels=disorders,
            cmap=cmap_seq, linewidth=.5)
plt.tight_layout()
plt.savefig(path+'figures/heatmap_dominance_enigma.eps')

plt.ion()
plt.figure()
plt.bar(np.arange(len(disorders)), np.sum(dominance, axis=1),
        tick_label=disorders)
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.savefig(path+'figures/bar_dominance_enigma.eps')

# plot cross validation
fig, (ax1, ax2) = plt.subplots(2, 1)
sns.violinplot(data=train_metric, ax=ax1)
sns.violinplot(data=test_metric, ax=ax2)
ax1.set(ylabel='train set correlation', ylim=(-1, 1))
ax2.set_xticklabels(disorders, rotation=90)
ax2.set(ylabel='test set correlation', ylim=(-1, 1))
plt.tight_layout()
plt.savefig(path+'figures/violin_enigma_cv.eps')

