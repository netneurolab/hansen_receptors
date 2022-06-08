# -*- coding: utf-8 -*-
"""
Figure 5: Receptor x Cognition PLS analysis

Note: to load pls_result:
pls_result = pyls.load_results(path+'results/pls_result.hdf5')
"""
import numpy as np
import pandas as pd
import pyls
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from netneurotools import datasets, stats, plotting
from scipy.stats import zscore, pearsonr, ttest_ind
from scipy.spatial.distance import squareform, pdist
from nilearn.datasets import fetch_atlas_schaefer_2018

def pls_cv_distance_dependent(X, Y, coords, trainpct=0.75, lv=0,
                              testnull=False, spins=None, nspins=1000):
    """
    Distance-dependent cross validation.

    Parameters
    ----------
    X : (n, p1) array_like
        Input data matrix. `n` is the number of brain regions.
    Y : (n, p2) array_like
        Input data matrix. `n` is the number of brain regions.
    coords : (n, 3) array_like
        Region (x,y,z) coordinates. `n` is the number of brain regions.
    trainpct : float 
        Percent observations in train set. 0 < trainpct < 1.
        Default = 0.75.
    lv : int
        Index of latent variable to cross-validate. Default = 0.
    testnull : Boolean
        Whether to calculate and return null mean test-set correlation.
    spins : (n, nspins) array_like
        Spin-test permutations. Required if testnull=True.
    nspins : int
        Number of spin-test permutations. Only used if testnull=True

    Returns
    -------
    train : (nplit, ) array
        Training set correlation between X and Y scores.me
    test : (nsplit, ) array
        Test set correlation between X and Y scores.

    """

    X = np.array(X)
    Y = np.array(Y)

    nnodes = len(coords)
    train = np.zeros((nnodes, ))
    test = np.zeros((nnodes, ))
    
    P = squareform(pdist(coords, metric="euclidean"))

    for k in range(nnodes):

        # distance from a node to all others
        distances = P[k, :]
        idx = np.argsort(distances)
        
        train_idx = idx[:int(np.floor(trainpct * nnodes))]
        test_idx = idx[int(np.floor(trainpct * nnodes)):]

        Xtrain = X[train_idx, :]
        Xtest = X[test_idx, :]
        Ytrain = Y[train_idx, :]
        Ytest = Y[test_idx, :]

        # pls analysis
        train_result = pyls.behavioral_pls(Xtrain, Ytrain, n_boot=0, n_perm=0, test_split=0)

        train[k], _ = pearsonr(train_result["x_scores"][:, lv],
                               train_result["y_scores"][:, lv])
        # project weights, correlate predicted scores in the test set
        test[k], _ = pearsonr(Xtest @ train_result["x_weights"][:, lv],
                              Ytest @ train_result["y_weights"][:, lv])

    # if testnull=True, get distribution of mean null test-set correlations.
    if testnull:
        print("Running null test-set correlations, will take time")

        testnull = np.zeros((nspins, nnodes))
        trainnull = np.zeros((nspins, nnodes))

        for k in range(nspins):  # will take a while
            print('test null iteration ' + str(k) + '/' + str(nspins))
            tr, te, _, _ = pls_cv_distance_dependent(X, Y[spins[:, k], :], coords)
            testnull[k, :]  = te
            trainnull[k, :] = tr
    else:
        testnull = None
        trainnull = None

    return train, test, testnull, trainnull

"""
set-up
"""
scale = 'scale100'
path = 'C:/Users/justi/OneDrive - McGill University/MisicLab/proj_receptors/\
github/hansen_receptors/'

# set up parcellation
schaefer = fetch_atlas_schaefer_2018(n_rois=100)
nnodes = len(schaefer['labels'])
coords = np.genfromtxt(path+'data/schaefer/coordinates/Schaefer_100_centres.txt')[:, 1:]
hemiid = np.zeros((nnodes, ))
hemiid[:int(nnodes/2)] = 1
nspins = 10000
spins = stats.gen_spinsamples(coords, hemiid, n_rotate=nspins, seed=1234)

# load receptor data
receptor_data = np.genfromtxt(path+'results/receptor_data_'+scale+'.csv',
                              delimiter=',')
receptor_names = np.load(path+'data/receptor_names_pet.npy')

# load neurosynth data
neurosynth = pd.read_csv(path+'data/neurosynth/ns_'+scale+'.csv',
                         delimiter=',')
neurosynth = neurosynth.drop(columns='Unnamed: 0')

# mesulam laminar classification
# from 1 --> 4 : paralimbic, heteromodal, unimodal, idiotypic
mesulam_mapping = np.genfromtxt(path+'data/mesulam_'+scale+'.csv').astype(int)

# colourmaps
cmap = np.genfromtxt(path+'data/colourmap.csv', delimiter=',')
cmap_div = ListedColormap(cmap)
cmap_seq = ListedColormap(cmap[128:, :])


"""
behavioural PLS
"""
X = zscore(receptor_data)
Y = zscore(neurosynth)

pls_result = pyls.behavioral_pls(X, Y, n_boot=nspins, n_perm=nspins, permsamples=spins,
                                 test_split=0, seed=1234)
pyls.save_results(path+'results/pls_result.hdf5', pls_result)

train, test, testnull, trainnull = pls_cv_distance_dependent(X, Y, coords, testnull=True,
                                        spins=spins, nspins=1000)
lv = 0  # latent variable
np.save(path+'results/pls_train.npy', train)
np.save(path+'results/pls_test.npy', test)
np.save(path+'results/pls_testnull.npy', testnull)

# covariance explained
# NOTE: `pls_result['permres']['perm_singval'] comes from locally editing
# pyls.base.py run_pls(self, X, Y) to return the `d_perm` variable.
# Also note that for this edit to work, the structures.PLSPermResult
# object needs to be updated to allow the `d_perm` to be returned.

cv = pls_result["singvals"]**2 / np.sum(pls_result["singvals"]**2)
null_singvals = pls_result['permres']['perm_singval']
cv_spins = null_singvals**2 / sum(null_singvals**2)
p = (1+sum(null_singvals[lv, :] > pls_result["singvals"][lv]))/(1+nspins)

# plot covariance explained
plt.ion()
plt.figure()
plt.boxplot(cv_spins.T * 100, positions=range(len(receptor_names)))
plt.scatter(range(len(receptor_names)), cv*100, s=80)
plt.ylabel("percent covariance accounted for")
plt.xlabel("latent variable")
plt.title('PLS' + str(lv) + ' cov exp = ' + str(cv[lv])[:5]
          + ', pspin = ' + str(p)[:5])
plt.tight_layout()
plt.savefig(path+'figures/schaefer100/scatter_pls_var_exp.eps')

# plot score correlation
plt.ion()
plt.figure()
sns.regplot(x=pls_result['x_scores'][:, lv], y=pls_result['y_scores'][:, lv],
            scatter=False)
plt.scatter(pls_result['x_scores'][:, lv], pls_result['y_scores'][:, lv],
            c = mesulam_mapping)
plt.xlabel('receptor scores')
plt.ylabel('cognitive term scores')
plt.savefig(path+'figures/schaefer100/scatter_scores.eps')

# plot scores on the surface
plt.ion()
annot = datasets.fetch_schaefer2018('fsaverage')['100Parcels7Networks']
brain = plotting.plot_fsaverage(data=pls_result["x_scores"][:, lv],
                                lhannot=annot.lh, rhannot=annot.rh,
                                colormap=cmap_div,
                                vmin=-np.max(np.abs(pls_result["x_scores"][:, lv])),
                                vmax=np.max(np.abs(pls_result["x_scores"][:, lv])),
                                views=['lat', 'med'],
                                data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/schaefer100/surface_pls_xscores.eps')

brain = plotting.plot_fsaverage(data=pls_result["y_scores"][:, lv],
                                lhannot=annot.lh, rhannot=annot.rh,
                                colormap=cmap_div,
                                vmin=-np.max(np.abs(pls_result["y_scores"][:, lv])),
                                vmax=np.max(np.abs(pls_result["y_scores"][:, lv])),
                                views=['lat', 'med'],
                                data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/schaefer100/surface_pls_yscores.eps')

# plot cross-validation
testnull =  np.genfromtxt(path+'results/pls_testnull.csv', delimiter=',')
emp = pearsonr(pls_result['x_scores'][:, lv], pls_result['y_scores'][:, lv])[0]
fig, ax = plt.subplots(figsize=(4, 4))
sns.boxplot(data=[train, test, np.mean(testnull, axis=0)], ax=ax)
# ax.plot(0, emp, 'o', c='red')  # full model correlation
# ax.plot(1, np.mean(test), 'o', c='red')  # mean test set correlation
ax.set_xticklabels(['train', 'test', 'null'])
ax.set_ylabel('score correlation')
plt.savefig(path+'figures/schaefer100/boxplot_pls_cv.eps')


"""
loadings
"""
# plot receptor loadings
# I'm lazy so just going to flip x and y in pls to get x CI
xload = pyls.behavioral_pls(Y, X, n_boot=10000, n_perm=0, test_split=0)
err = (xload["bootres"]["y_loadings_ci"][:, lv, 1]
      - xload["bootres"]["y_loadings_ci"][:, lv, 0]) / 2
sorted_idx = np.argsort(xload["y_loadings"][:, lv])
plt.figure()
plt.ion()
plt.bar(range(len(receptor_names)), xload["y_loadings"][sorted_idx, lv],
        yerr=err[sorted_idx])
plt.xticks(range(len(receptor_names)),
           labels=[receptor_names[i] for i in sorted_idx],
           rotation='vertical')
plt.ylabel("Receptor loadings")
plt.tight_layout()
plt.savefig(path+'figures/schaefer100/bar_pls_rload.eps')

# plot term loadings
err = (pls_result["bootres"]["y_loadings_ci"][:, lv, 1]
      - pls_result["bootres"]["y_loadings_ci"][:, lv, 0]) / 2
relidx = (abs(pls_result["y_loadings"][:, lv]) - err) > 0  # CI doesnt cross 0
sorted_idx = np.argsort(pls_result["y_loadings"][relidx, lv])
plt.figure(figsize=(10, 5))
plt.ion()
plt.bar(np.arange(sum(relidx)), np.sort(pls_result["y_loadings"][relidx, lv]),
        yerr=err[relidx][sorted_idx])
plt.xticks(np.arange(sum(relidx)), labels=neurosynth.columns[relidx][sorted_idx],
           rotation='vertical')
plt.ylabel("Cognitive term loadings")
plt.tight_layout()
plt.savefig(path+'figures/schaefer100/bar_pls_tload.eps')

# compare loadings across receptor classes
exc = ['5HT2a', '5HT4', '5HT6', 'D1', 'mGluR5', 'A4B2', 'M1', 'NMDA']
inh = ['5HT1a', '5HT1b', 'CB1', 'D2', 'GABAa', 'H3', 'MOR']
mami = ['5HT1a', '5HT1b', '5HT2a', '5HT4', '5HT6', '5HTT', 'D1',
        'D2', 'DAT', 'H3', 'NET']
nmami = list(set(receptor_names) - set(mami))
metab = ['5HT1a', '5HT1b', '5HT2a', '5HT4', '5HT6', 'CB1', 'D1',
         'D2', 'H3', 'M1', 'mGluR5', 'MOR']
iono = ['A4B2', 'GABAa', 'NMDA']
gspath = ['5HT4', '5HT6', 'D1']
gipath = ['CB1', 'D2', 'H3', '5HT1a', '5HT1b', 'MOR']
gqpath = ['5HT2a', 'mGluR5', 'M1']

i_exc = np.array([list(receptor_names).index(i) for i in exc])
i_inh = np.array([list(receptor_names).index(i) for i in inh])
i_mami = np.array([list(receptor_names).index(i) for i in mami])
i_nmami = np.array([list(receptor_names).index(i) for i in nmami])
i_metab =  np.array([list(receptor_names).index(i) for i in metab])
i_iono = np.array([list(receptor_names).index(i) for i in iono])
i_gs = np.array([list(receptor_names).index(i) for i in gspath])
i_gi = np.array([list(receptor_names).index(i) for i in gipath])
i_gq = np.array([list(receptor_names).index(i) for i in gqpath])

classes = [[i_exc, i_inh], [i_mami, i_nmami],
           [i_metab, i_iono], [i_gs, i_gi, i_gq]]
class_names = [['exc', 'inh'], ['monoamine', 'not'],
               ['metabotropic', 'ionotropic'], ['gs', 'gi', 'gq']]
plt.ion()
fig, axs = plt.subplots(1, 4, figsize=(15, 3))
axs = axs.ravel()
for i in range(len(classes)):
    print(class_names[i])
    d = [xload["y_loadings"][:, lv][classes[i][j]] for j in range(len(classes[i]))]
    print(ttest_ind(d[0], d[1]))
    if len(d) > 2:
        print(ttest_ind(d[0], d[2]))
        print(ttest_ind(d[1], d[2]))
    sns.violinplot(data=d, inner=None, color=".8", ax=axs[i])
    sns.stripplot(data=d, ax=axs[i])
    axs[i].set_xticklabels(class_names[i])
    axs[i].set_ylabel('loadings')
plt.tight_layout()
plt.savefig(path+'figures/schaefer100/stripplot_pls_rclasses.eps')