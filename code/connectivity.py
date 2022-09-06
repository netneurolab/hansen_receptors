# -*- coding: utf-8 -*-
"""
Figure 3: structure-function relationships
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from netneurotools import datasets, stats, plotting, metrics
from scipy.stats import zscore, pearsonr
import seaborn as sns
from scipy.spatial.distance import squareform, pdist
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.utils.validation import check_random_state
from nilearn.datasets import fetch_atlas_schaefer_2018
from statsmodels.stats.multitest import multipletests


def get_reg_r_sq(X, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    yhat = lin_reg.predict(X)
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    adjusted_r_squared = 1 - (1 - r_squared) * \
        (len(y) - 1) / (len(y) - X.shape[1] - 1)
    return adjusted_r_squared, SS_Residual


def exponential(x, a, b, c):
    return a * np.exp(b * x) + c


def regress_dist(x, eu_distance, pars):
    return x - exponential(eu_distance, pars[0], pars[1], pars[2])


def match_length_degree_distribution(data, eu_distance, nbins=10, nswap=None, seed=None):
    """
    Takes a weighted, symmetric connectivity matrix `data` and Euclidean/fiber 
    length matrix `distance` and generates a randomized network with:
        1. exactly the same degree sequence
        2. approximately the same edge length distribution
        3. exactly the same edge weight distribution
        4. approximately the same weight-length relationship

    Parameters
    ----------
    data : (N, N) array-like
        weighted or binary symmetric connectivity matrix.
    distance : (N, N) array-like
        symmetric distance matrix.
    nbins : int
        number of distance bins (edge length matrix is performed by swapping
        connections in the same bin). Default = 10.
    nswap : int
        total number of edge swaps to perform. Recommended = nnodes * 20.
        Default = None.

    Returns
    -------
    data : (N, N) array-like
        binary rewired matrix
    W : (N, N) array-like
        weighted rewired matrix
        
    Reference
    ---------
    Betzel, R. F., Bassett, D. S. (2018) Specificity and robustness of long-distance
    connections in weighted, interareal connectomes. PNAS.

    """
    rs = check_random_state(seed)

    nnodes = len(data)             # number of nodes
    
    if nswap is None:
        nswap = nnodes*20          # set default number of swaps
    
    mask = data != 0               # nonzero elements
    mask = np.triu(mask, 1)        # keep upper triangle only
    weights = data[mask]           # values of edge weights
    distances = eu_distance[mask]  # values of edge lengths
    Jdx = np.argsort(distances)    # indices to sort distances in ascending order
    
    bins = np.linspace(min(eu_distance[eu_distance != 0]),
                       max(eu_distance[eu_distance != 0]),
                       nbins+1)  # length/distance of bins
    bins[-1] += 1
    B = np.zeros((nnodes, nnodes, nbins))  # initiate 3D stack of bins
    for k in range(nbins):
        # element is k+1 if the distance falls within the bin, 0 otherwise
        B[:, :, k] = np.logical_and(eu_distance >= bins[k],
                                   eu_distance < bins[k + 1]) * (k + 1)
    # matrix of distance bins
    Bsum = np.sum(B, axis=2)
    
    tmp = np.triu((data != 0)*Bsum, 1)
    row_idx, col_idx = tmp.nonzero()  # indices of edges
    vals = tmp[row_idx, col_idx]
    nedges = len(row_idx)  # number of edges
    iswap = 0             # swap counter
    
    while iswap < nswap:
        myEdge = rs.randint(nedges)   # get a random edge index
        myEdge_row = row_idx[myEdge]  # row idx of edge
        myEdge_col = col_idx[myEdge]  # col idx of edge
        myEdge_bin = vals[myEdge]     # bin of edge
        
        # get indices that can be swapped
        indkeep = (row_idx != myEdge_row) & (row_idx != myEdge_col) \
                  & (col_idx != myEdge_row) & (col_idx != myEdge_col)

        row_idx_keep = row_idx[indkeep]
        col_idx_keep = col_idx[indkeep]
        
        bins_keep = vals[indkeep]  # bins of possible swaps
        
        edge_row = myEdge_row*nnodes + row_idx_keep # edge indices
        edge_row_bins = Bsum[np.unravel_index(edge_row, Bsum.shape)] # matlab-style linear indexing
        edge_col = myEdge_col*nnodes + col_idx_keep # other set of edge indices
        edge_col_bins = Bsum[np.unravel_index(edge_col, Bsum.shape)] 
        
        # get good list of indices
        idx1 = np.logical_and(myEdge_bin == edge_row_bins,
                              bins_keep == edge_col_bins)
        # get other set of good indices
        idx2 = np.logical_and(myEdge_bin == edge_col_bins,
                              bins_keep == edge_row_bins)
        # full set
        goodidx = np.logical_or(idx1, idx2)
        
        # update the indices to keep
        row_idx_keep = row_idx_keep[goodidx]
        col_idx_keep = col_idx_keep[goodidx]
        
        # update the edge indices
        edge_row = myEdge_row*nnodes + row_idx_keep
        edge_col = myEdge_col*nnodes + col_idx_keep
        
        data_row = data[np.unravel_index(edge_row, data.shape)]
        data_col = data[np.unravel_index(edge_col, data.shape)]
        
        # find missing edges
        ind = np.where(np.logical_and(data_row == 0,
                                      data_col == 0).astype(int))[0]
        
        if len(ind) > 0:  # if there is a missing edge
            
            # choose a random swap
            random_swap = ind[rs.randint(len(ind))]
            
            # do the swap
            row_idx_keep = row_idx_keep[random_swap]
            col_idx_keep = col_idx_keep[random_swap]
            
            data[myEdge_row, myEdge_col] = 0
            data[myEdge_col, myEdge_row] = 0
            
            data[row_idx_keep, col_idx_keep] = 0
            data[col_idx_keep, row_idx_keep] = 0
            
            data[myEdge_row, row_idx_keep] = 1
            data[row_idx_keep, myEdge_row] = 1
            
            data[myEdge_col, col_idx_keep] = 1
            data[col_idx_keep, myEdge_col] = 1
            
            other_edge = np.where(indkeep)[0]
            other_edge = other_edge[goodidx]
            other_edge = other_edge[random_swap]
            
            row_idx[myEdge] = min(myEdge_row, row_idx_keep)
            col_idx[myEdge] = max(myEdge_row, row_idx_keep)
            
            row_idx[other_edge] = min(myEdge_col, col_idx_keep)
            col_idx[other_edge] = max(myEdge_col, col_idx_keep)
            
            vals[myEdge] = Bsum[myEdge_row, row_idx_keep]
            vals[other_edge] = Bsum[myEdge_col, col_idx_keep]
            
            iswap += 1
            # if iswap % 100 == 0:
            #     print(iswap)
    
    d = eu_distance[np.where(np.triu(data, 1))]  # get distances where edges are
    jdx = np.argsort(d)                      # sort distances (ascending)
    W = np.zeros((nnodes, nnodes))           # output matrix
    # add weights
    W[np.where(np.triu(data,1))[0][jdx],
      np.where(np.triu(data,1))[1][jdx]] = weights[Jdx]
    
    return data, W


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
            train_metric.append(get_reg_r_sq(X[train_idx, :], y[train_idx])[0])

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


def corr_spin(x, y, spins, nspins):
    rho, _ = pearsonr(x, y)
    null = np.zeros((nspins,))

    # null correlation
    for i in range(nspins):
        null[i], _ = pearsonr(x[spins[:, i]], y)

    pval = (1 + sum(abs((null - np.mean(null))) >
                    abs((rho - np.mean(null))))) / (nspins + 1)
    return rho, pval


def get_perm_p(emp, null):
    return (1 + sum(abs(null - np.mean(null))
                    > abs(emp - np.mean(null)))) / (len(null) + 1)



"""
set-up
"""

path = 'C:/Users/justi/OneDrive - McGill University/MisicLab/proj_receptors/\
github/hansen_receptors/'

# get the parcellation, coordinates, etc
scale = 'scale100'

schaefer = fetch_atlas_schaefer_2018(n_rois=100)
nnodes = len(schaefer['labels'])
coords = np.genfromtxt(path+'data/schaefer/coordinates/Schaefer_100_centres.txt')[:, 1:]
hemiid = np.zeros((nnodes, ))
hemiid[:int(nnodes/2)] = 1
nspins = 10000
spins = stats.gen_spinsamples(coords, hemiid, n_rotate=nspins, seed=1234)
eu = squareform(pdist(coords, metric='euclidean'))

# load and process structure/function data
sc = np.load(path+'data/schaefer/sc_binary.npy')
sc_weighted = np.load(path+'data/schaefer/sc_weighted.npy')
fc = np.load(path+'data/schaefer/fc_weighted.npy')

# load the receptor data
receptor_data = np.genfromtxt(path+'results/receptor_data_'+scale+'.csv', delimiter=',')
receptor_similarity = np.corrcoef(zscore(receptor_data))
mask = np.triu(np.ones(nnodes), 1) > 0

# colourmaps
cmap = np.genfromtxt(path+'data/colourmap.csv', delimiter=',')
cmap_div = ListedColormap(cmap)
cmap_seq = ListedColormap(cmap[128:, :])
cmap_blue = ListedColormap(np.flipud(cmap[:128, :]))

"""
Connected vs not connected
"""

# set up dictionary with connected vs not connected receptor similarity
d = dict({'connected': receptor_similarity[mask][np.where(sc[mask] == 1)],
          'not connected': receptor_similarity[mask][np.where(sc[mask] == 0)]})
df_sc = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))

emp = np.mean(df_sc['connected']) - np.mean(df_sc['not connected'])

null = np.zeros((nspins, 1))
for i in range(nspins):
    sc_rewired, _ = match_length_degree_distribution(sc, eu, 10, nnodes*20)
    null[i] = np.mean(receptor_similarity[mask]
                      [np.where(sc_rewired[mask] == 1)]) \
              - np.mean(receptor_similarity[mask]
                        [np.where(sc_rewired[mask] == 0)])

np.save(path+'results/sc_edge-degree-preserving_null.npy', null)
pval_sc = (1 + np.sum(np.abs((null - np.mean(null)))
                      >= abs((emp - np.mean(null))))) / (nspins + 1)


"""
Within vs between network
"""

rsn_mapping = []
for row in range(len(schaefer['labels'])):
    rsn_mapping.append(schaefer['labels'][row].decode('utf-8').split('_')[2])
rsn_mapping = np.array(rsn_mapping)

# is an edge within or between networks?
withbet = np.zeros(fc.shape)
for k in range(nnodes):
    for j in range(nnodes):
        if rsn_mapping[k] == rsn_mapping[j]:
            withbet[k, j] = 1

# dictionary of within vs between receptor similarity
d = dict({'within': receptor_similarity[mask][np.where(withbet[mask] == 1)],
          'between': receptor_similarity[mask][np.where(withbet[mask] == 0)]})
df_fc = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))

# spin null
emp = np.mean(df_fc['within']) - np.mean(df_fc['between'])
nspins = 1000
null = np.zeros([nspins, 1])
for i in range(nspins):
    rsn_null = rsn_mapping[spins[:, i]]
    withbet_null = np.zeros(fc.shape)
    for k in range(nnodes):
        for j in range(nnodes):
            if rsn_null[k] == rsn_mapping[j]:
                withbet_null[k, j] = 1
    fc_null = dict({'within': receptor_similarity[mask]
                   [np.where(withbet_null[mask] == 1)],
                   'between': receptor_similarity[mask]
                   [np.where(withbet_null[mask] == 0)]})
    null[i] = np.mean(fc_null['within']) - np.mean(fc_null['between'])

pval_fc = (1 + np.sum(np.abs((null - np.mean(null)))
                      >= abs((emp - np.mean(null))))) / (nspins + 1)

# exponential fits
p0 = [1, -0.05, -0.1]  # initial parameter guesses
pars, _ = curve_fit(exponential, eu[mask], receptor_similarity[mask], p0=p0)
rs_reg = regress_dist(receptor_similarity[mask], eu[mask], pars)
parsF, _ = curve_fit(exponential, eu[mask], fc[mask], p0=p0)
fc_reg = regress_dist(fc[mask], eu[mask], parsF)
nosc_idx = sc_weighted[mask] != 0
parsS, _ = curve_fit(exponential, eu[mask][nosc_idx],
                   sc_weighted[mask][nosc_idx], p0=p0)
sc_reg = regress_dist(sc_weighted[mask][nosc_idx],
                      eu[mask][nosc_idx], parsS)

"""
plot
"""

inds = plotting.sort_communities(receptor_similarity, rsn_mapping)
bounds = plotting._grid_communities(rsn_mapping)
bounds[0] += 0.2
bounds[-1] -= 0.2

plt.ion()

# sc
fig, ax = plt.subplots()
sns.heatmap(data=sc_weighted[np.ix_(inds, inds)], cmap=cmap_blue, vmin=0,
            ax=ax, cbar=True, square=True,
            xticklabels=False, yticklabels=False)
for n, edge in enumerate(np.diff(bounds)):
    ax.add_patch(patches.Rectangle((bounds[n], bounds[n]),
                                    edge, edge, fill=False, linewidth=2,
                                    edgecolor='black'))
plt.savefig(path+'figures/schaefer100/heatmap_sc_weighted.eps')

# fc
fig, ax = plt.subplots()
sns.heatmap(data=fc[np.ix_(inds, inds)], cmap=cmap_div,
            vmin=-1, vmax=1,
            ax=ax, cbar=True, square=True,
            xticklabels=False, yticklabels=False)
for n, edge in enumerate(np.diff(bounds)):
    ax.add_patch(patches.Rectangle((bounds[n], bounds[n]),
                                    edge, edge, fill=False, linewidth=2,
                                    edgecolor='black'))
plt.savefig(path+'figures/schaefer100/heatmap_fc.eps')

# everything else
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
sns.boxplot(data=df_sc, ax=ax1)
sns.boxplot(data=df_fc, ax=ax2)
sns.regplot(sc_reg, rs_reg[nosc_idx], marker='.', scatter_kws={'s' : 5}, ax=ax3)
sns.regplot(fc_reg, rs_reg, marker='.', scatter_kws={'s' : 5}, ax=ax4)
ax1.set_ylabel('receptor similarity')
ax1.set_title(['p = ' + str(pval_sc)[:5]])
ax2.set_title(['p = ' + str(pval_fc)[:5]])
ax3.set_xlabel('weighted sc')
ax3.set_ylabel('receptor similarity')
r, p = pearsonr(rs_reg[nosc_idx], sc_reg)
ax3.set_title(['r = ' + str(r)[:4] + ', p = ' + str(p)[:5]])
ax4.set_xlabel('functional connectivity')
ax4.set_ylabel('receptor similarity')
r, p = pearsonr(rs_reg, fc_reg)
ax4.set_title(['r = ' + str(r)[:4] + ', p = ' + str(p)[:5]])
plt.tight_layout()
plt.savefig(path+'figures/schaefer100/connectivity.eps')

"""
Structure-Function coupling
"""

co = metrics.communicability_wei(sc_weighted)
# co = metrics.communicability_bin(sc)
# mi = similarity.matching_ind(sc_weighted)[2]
# cs = cosine_similarity(sc_weighted)

rsq_sc = np.zeros([nnodes, ])
rsq_r = np.zeros([nnodes, ])
train = np.zeros([nnodes, nnodes, 2])
test = np.zeros(train.shape)
rnull = np.zeros([nnodes, nspins])

for i in range(nnodes):
    print(i)
    y = fc[:, i]
    x1 = co[:, i]
    x2 = receptor_similarity[:, i]

    x_sc = zscore(x1).reshape(-1, 1)
    x_r = zscore(np.stack((x1, x2), axis=1))
    rsq_sc[i], res_sc = get_reg_r_sq(x_sc, y)
    rsq_r[i], res_r = get_reg_r_sq(x_r, y)
    train[i, :, 0], test[i, :, 0] = cv_slr_distance_dependent(x_sc, y, coords, metric='corr')
    train[i, :, 1], test[i, :, 1] = cv_slr_distance_dependent(x_r, y, coords, metric='corr')

    for s in range(nspins):
        x_rnull = zscore(np.stack((x1, x2[spins[:, s]]), axis=1))
        rnull[i, s], _ = get_reg_r_sq(x_rnull, y)

rpvals = np.zeros([nnodes, ])
for i in range(nnodes):
    rpvals[i] = (1 + sum(rnull[i, :] > rsq_r[i]))/(nspins + 1)
rpvals = multipletests(rpvals, method='fdr_bh')[1]
np.save(path+'results/rsq_null.npy', rnull)
np.save(path+'results/scfc_coupling_train.npy', train)
np.save(path+'results/scfc_coupling_test.npy', test)

plt.ion()
fig, ax = plt.subplots()
plt.scatter(rsq_sc, rsq_r, c=(rpvals < 0.05).astype(int))
plt.plot(rsq_sc, rsq_sc, 'k-', linewidth=.5)
plt.xlabel('Rsq from SC only')
plt.ylabel('Rsq from SC + Receptors')
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.savefig(path+'figures/schaefer100/scatter_scfc_coupling.eps')

annot = datasets.fetch_schaefer2018('fsaverage')['100Parcels7Networks']
brain = plotting.plot_fsaverage(data=rsq_r,
                                lhannot=annot.lh, rhannot=annot.rh,
                                colormap=cmap_seq, vmin=0, vmax=0.60,
                                views=['lat', 'med'],
                                data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/schaefer100/surface_rsq_r.eps')

brain = plotting.plot_fsaverage(data=rsq_sc,
                                lhannot=annot.lh, rhannot=annot.rh,
                                colormap=cmap_seq, vmin=0, vmax=0.60,
                                views=['lat', 'med'],
                                data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/schaefer100/surface_rsq_sc.eps')

brain = plotting.plot_fsaverage(data=rsq_r - rsq_sc,
                                lhannot=annot.lh, rhannot=annot.rh,
                                colormap=cmap_seq,
                                vmin=0, vmax=max(rsq_r - rsq_sc),
                                views=['lat', 'med'],
                                data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/schaefer100/surface_scfc_coupling.eps')

brain = plotting.plot_fsaverage(data=np.mean(np.squeeze(test[:, :, 1]), axis=1),
                                lhannot=annot.lh, rhannot=annot.rh,
                                colormap=cmap_seq,
                                vmin=0, vmax=np.max(np.mean(test[:, :, 1], axis=1)),
                                views=['lat', 'med'],
                                data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/schaefer100/surface_rsq_test_r.eps')

brain = plotting.plot_fsaverage(data=np.mean(np.squeeze(test[:, :, 0]), axis=1),
                                lhannot=annot.lh, rhannot=annot.rh,
                                colormap=cmap_seq,
                                vmin=0, vmax=np.max(np.mean(test[:, :, 0], axis=1)),
                                views=['lat', 'med'],
                                data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/schaefer100/surface_rsq_test_sc.eps')