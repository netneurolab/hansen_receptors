# -*- coding: utf-8 -*-
"""
Autoradiography 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns
from netneurotools import datasets, metrics, stats, plotting
from scipy.stats import zscore, pearsonr
from scipy.optimize import curve_fit
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import squareform, pdist
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_random_state
from sklearn.decomposition import PCA
from nilearn.datasets import fetch_atlas_schaefer_2018
import pyls
from statsmodels.stats.multitest import multipletests


def make_autorad_schaefer100(autorad_zilles44):
    
    autorad_schaefer100 = np.zeros((50, autorad_zilles44.shape[1]))
    # mapping between zilles and schaefer (done manually)
    zilles44_to_schaefer100 = \
        [[21],  # area '36' // 0
         [9],   # area 'V3v' // 1
         [6],   # area 'V2v' // 2
         [4, 6],   # area 'V1', 'V2v'  // 3
         [4, 5, 6],   # area 'V1', 'V2v', 'V2d' // 4
         [4],   # area 'V1' // 5
         [np.nan],  # // 6
         [8, 7],   # area 'V3d', 'V3a' // 7
         [8],   # area 'V3d' // 8
         [17],  # area '42' // 9
         [16],  # area '41' // 10
         [np.nan], # // 11
         [0, 3, 2],  # area '1', '3b', '3a' \\ 12
         [0, 3, 2],  # area '1', '3b', '3a' \\ 13
         [0, 3, 2],  # area '1', '3b', '3a'\\ 14
         [14],  # area '37L' \\ 15
         [27, 1],  # area 'PFt', '2' \\ 16
         [np.nan],  # // 17
         [1],  # area '2' // 18
         [24],  # area '5M' // 19
         [23],  # area '5L' // 20
         [0, 34],  # area '1', '6' // 21
         [35],  # area '8' // 22
         [28, 27],  # area 'PFm', 'PFt' // 23
         [np.nan],  # insula // 24
         [41],  # area '45' // 25
         [36],  # area '9' // 26
         [29],  # area '24' // 27
         [24],  # area '5M' // 28
         [35],  # area '8' // 29
         [39],  # area '11' // 30
         [22],  # area '38' // 31
         [18],  # area '20' // 32
         [28, 27],  # area 'PFm' 'PFt' // 33
         [42],  # area '46' // 34
         [7],  # area 'V3A' // 35
         [31],  # area '23' (and 31?) // 36
         [20, 19],  # area '22', '21' // 37
         [19, 20],  # area '21', '22' // 38
         [19, 20],  # area '21', '22' // 39
         [25, 26],  # area 'PGa', 'PGp' // 40
         [43],  # area '47' // 41
         [41, 43, 42],  # area '45', '47', '46' // 42
         [29, 30, 38],  # area '24', '32', '10M' // 43
         [37],  # area '10L' // 44
         [36],  # area '9' // 45
         [35, 36],  # area '8', '9' // 46
         [35],  # area '8' // 47
         [32],  # area '31' // 48
         [32]]  # area '31' // 49 

    for n in range(autorad_schaefer100.shape[0]):
        if np.isnan(zilles44_to_schaefer100[n][0]):
            autorad_schaefer100[n, :] = np.nan
        elif len(zilles44_to_schaefer100[n]) == 1:
            autorad_schaefer100[n, :] = autorad_zilles44[zilles44_to_schaefer100[n], :]
        elif len(zilles44_to_schaefer100[n]) > 1:
            autorad_schaefer100[n, :] = np.mean(autorad_zilles44[zilles44_to_schaefer100[n], :], axis=0).T

    return autorad_schaefer100


def make_autorad_cammoun033(autorad_zilles44):
    # region indeces associated with more than one dk region
    duplicate = [20, 21, 28, 29, 30, 32, 34, 39]

    # mapping from 44 brodmann areas + 7 duplicate regions to dk left hem
    # manually done, comparing anatomical regions to one another
    # originally written in matlab and too lazy to change indices hence the -1
    # the index refers to the cammoun scale033 structure name
    mapping = np.array([57, 57, 57, 57, 63, 62, 65, 62, 64, 65, 64, 66, 66,
                        66, 66, 66, 74, 74, 70, 71, 72, 73, 67, 68, 69, 52,
                        52, 60, 60, 58, 58, 59, 53, 54, 53, 54, 55, 56, 61,
                        51, 51, 50, 49, 49, 44, 44, 45, 42, 47, 46, 48, 43])
    mapping = mapping - min(mapping)  # python indexing

    rep = np.ones((autorad_zilles44.shape[0], ), dtype=int)  # duplicate regions
    rep[duplicate] = 2
    autorad_zilles44 = np.repeat(autorad_zilles44, rep, 0)

    # convert to dk
    n_dknodes = max(mapping) + 1  # number of nodes in dk atlas (left hem only)

    u = np.unique(mapping)
    autorad_cammoun033 = np.zeros((n_dknodes, autorad_zilles44.shape[1]))
    for i in range(len(u)):
        if sum(mapping == u[i]) > 1:
            autorad_cammoun033[u[i], :] = np.mean(autorad_zilles44[mapping == u[i], :], axis=0)
        else:
            autorad_cammoun033[u[i], :] = autorad_zilles44[mapping == u[i], :]
    return autorad_cammoun033
    

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


def get_perm_p(emp, null):
    return (1 + sum(abs(null - np.mean(null))
                    > abs(emp - np.mean(null)))) / (len(null) + 1)


def corr_perm(x, y, perms, nperms):
    rho = pearsonr(x, y)[0]
    null = np.zeros((nperms, ))
    for i in range(nperms):
        null[i] = pearsonr(x, y[perms[:, i]])[0]
    pval = get_perm_p(rho, null)
    return rho, pval
    

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
            if iswap % 100 == 0:
                print(iswap)
    
    d = eu_distance[np.where(np.triu(data, 1))]  # get distances where edges are
    jdx = np.argsort(d)                      # sort distances (ascending)
    W = np.zeros((nnodes, nnodes))           # output matrix
    # add weights
    W[np.where(np.triu(data,1))[0][jdx],
      np.where(np.triu(data,1))[1][jdx]] = weights[Jdx]
    
    return data, W


def exponential(x, a, b, c):
    return a * np.exp(b * x) + c


def regress_dist(x, eu_distance, pars):
    return x - exponential(eu_distance, pars[0], pars[1], pars[2])


def add_hem_for_plotting(data, nnodes_full):
    """
    add in some zeros where data is missing so that plotting is possible
    """
    newdata = np.zeros([nnodes_full, ])
    newdata[:len(data)] = data
    return newdata



"""
set-up
"""

path = 'C:/Users/justi/OneDrive - McGill University/MisicLab/proj_receptors/\
github/hansen_receptors/'

# load atlas
schaefer = fetch_atlas_schaefer_2018(n_rois=100)
annot = datasets.fetch_schaefer2018('fsaverage')['100Parcels7Networks']
nnodes = len(schaefer['labels'])
coords = np.genfromtxt(path+'data/schaefer/coordinates/Schaefer_100_centres.txt')[:, 1:]
hemiid = np.zeros((nnodes, ))
hemiid[:int(nnodes/2)] = 1
nspins = 10000
spins = stats.gen_spinsamples(coords, hemiid, n_rotate=nspins, seed=1234)
eu = squareform(pdist(coords, metric='euclidean'))

# load PET data
PET_data = np.genfromtxt(path+'results/receptor_data_scale100.csv', delimiter=',')

# load autorad data
autorad_zilles44 = np.load(path+'data/autoradiography/ReceptData.npy')
receptor_names = np.load(path+'data/autoradiography/ReceptorNames.npy')
autorad_schaefer100 = make_autorad_schaefer100(autorad_zilles44)
autorad_cammoun033 = make_autorad_cammoun033(autorad_zilles44)
goodidx = np.where(np.isnan(autorad_schaefer100[:, 0]) == False)[0]
badidx = np.where(np.isnan(autorad_schaefer100[:, 0]) == True)[0]
autorad_schaefer100 = zscore(autorad_schaefer100, nan_policy='omit')
nnodes = autorad_schaefer100.shape[0]
autorad_df = pd.DataFrame(data=autorad_schaefer100,
                          index=schaefer['labels'][:50],
                          columns=receptor_names)
perms = np.zeros((len(goodidx), nspins))
for i in range(nspins):
    perms[:, i] = np.random.permutation(len(goodidx))
perms = perms.astype(int)

# load sc fc
sc = np.load(path+'data/schaefer/sc_binary.npy')
sc_weighted = np.load(path+'data/schaefer/sc_weighted.npy')
fc = np.load(path+'data/schaefer/fc_weighted.npy')

# colourmaps
cmap = np.genfromtxt(path+'/data/colourmap.csv', delimiter=',')
cmap_div = ListedColormap(cmap)
cmap_seq = ListedColormap(cmap[128:, :])

# rsn networks for plotting
rsn_mapping = []
for row in range(len(schaefer['labels'])):
    rsn_mapping.append(schaefer['labels'][row].decode('utf-8').split('_')[2])
rsn_mapping = np.array(rsn_mapping)

"""
Receptor similarity
"""
plt.ion()

# correlating receptors
plt.figure()
sns.heatmap(data=autorad_df.corr(), cmap=cmap_div, vmin=-1, vmax=1,
            square=True, linewidths=.5, cbar=False)
plt.tight_layout()
plt.savefig(path+'figures/schaefer100/heatmap_autorad_recept_corr.eps')

# receptor similarity
bounds = plotting._grid_communities(rsn_mapping[goodidx])
bounds[0] += 0.2
bounds[-1] -= 0.2
fig, ax = plt.subplots()
sns.heatmap(data=np.corrcoef(autorad_schaefer100[goodidx, :]),
            cmap=cmap_div, vmin=-1, vmax=1, square=True, cbar=False,
            ax=ax, xticklabels=False, yticklabels=False, linewidths=.5)
for n, edge in enumerate(np.diff(bounds)):
    ax.add_patch(patches.Rectangle((bounds[n], bounds[n]),
                                    edge, edge, fill=False, linewidth=.5,
                                    edgecolor='black'))
plt.tight_layout()
plt.savefig(path+'figures/schaefer100/heatmap_autrad_receptor_similarity.eps')

# correlation between PET + autorad receptor similarity
PET_sim = np.corrcoef(zscore(PET_data)[goodidx, :])
autorad_sim = np.corrcoef(autorad_schaefer100[goodidx, :])
mask = np.triu(np.ones(autorad_sim.shape[0]), 1) > 0
r, p = pearsonr(autorad_sim[mask], PET_sim[mask])

plt.figure()
sns.regplot(autorad_sim[mask], PET_sim[mask], scatter=False)
plt.scatter(autorad_sim[mask], PET_sim[mask], s=5)
plt.xlabel('autoradiography receptor similarity')
plt.ylabel('PET receptor similarity')
plt.title('rho = ' + str(r)[:5] + ', p = ' + str(p)[:5])
plt.savefig(path+'figures/schaefer100/scatter_autorad_v_pet.eps')

# PC1
pca = PCA(n_components=1)
aut_pc1 = -np.squeeze(pca.fit_transform(autorad_schaefer100[goodidx, :]))  # flip sign to match PET PC1
pet_pc1 = np.squeeze(pca.fit_transform(zscore(PET_data)))
r, p = corr_perm(aut_pc1, pet_pc1[goodidx], perms, nspins)

brain = plotting.plot_fsaverage(data=add_hem_for_plotting(aut_pc1, nnodes*2),
                                lhannot=annot.lh, rhannot=annot.rh,
                                noplot=schaefer['labels'][badidx],
                                colormap=cmap_div,
                                vmin=-max(abs(aut_pc1)),
                                vmax=max(abs(aut_pc1)),
                                views=['lat', 'med'],
                                data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/schaefer100/surface_autorad_pc1.eps')

plt.figure()
sns.regplot(aut_pc1, pet_pc1[goodidx], scatter=False)
plt.scatter(aut_pc1, pet_pc1[goodidx])
plt.xlabel('autoradiography PC1sim')
plt.ylabel('PET PC1sim')
plt.title('rho = ' + str(r)[:5] + ', pspin = ' + str(p)[:5])
plt.savefig(path+'figures/schaefer100/scatter_autorad_v_pet_pc.eps')

"""
exponential
"""
p0 = [1, -0.05, -0.1]  # initial parameter guesses
pars, _ = curve_fit(exponential,
                    eu[np.ix_(goodidx, goodidx)][mask],
                    autorad_sim[mask], p0=p0)

fig, ax = plt.subplots()
ax.scatter(eu[np.ix_(goodidx, goodidx)][mask],
           autorad_sim[mask], s=5)
ax.plot(np.arange(10, 160), exponential(np.arange(10, 160),
                                         pars[0], pars[1], pars[2]), color='red')
ax.set_xlabel('euclidean distance')
ax.set_ylabel('receptor similarity')
ax.set_title('y = ' + str(pars[0])[:4]
          + ' * exp(' + str(pars[1])[:5]
          + ' * x) + '+ str(pars[2])[:5])
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.tight_layout()
plt.savefig(path+'figures/schaefer100/scatter_autorad_distance.eps')

"""
structure: connected vs not connected
"""

# set up dictionary with connected vs not connected receptor similarity
d = dict({'connected': autorad_sim[mask][np.where(sc[np.ix_(goodidx, goodidx)][mask] == 1)],
          'not connected': autorad_sim[mask][np.where(sc[np.ix_(goodidx, goodidx)][mask] == 0)]})
df_sc = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))

emp = np.mean(df_sc['connected']) - np.mean(df_sc['not connected'])

null = np.zeros((nspins, ))
for i in range(nspins):
    sc_rewired, _ = match_length_degree_distribution(sc,
                                                     eu,
                                                     10, nnodes*2*20)
    sc_rew = sc[np.ix_(goodidx, goodidx)]
    null[i] = np.mean(autorad_sim[mask]
                      [np.where(sc_rew[mask] == 1)]) \
              - np.mean(autorad_sim[mask]
                        [np.where(sc_rew[mask] == 0)])
np.save(path+'results/sc_edge-degree-preserving_null_autorad.npy', null)

pval_sc = (1 + np.sum(np.abs((null - np.mean(null)))
                      >= abs((emp - np.mean(null))))) / (nspins + 1)

"""
function: within vs between
"""
withbet = np.zeros((nnodes, nnodes))
for k in range(nnodes):
    for j in range(nnodes):
        if rsn_mapping[k] == rsn_mapping[j]:
            withbet[k, j] = 1
withbet = withbet[np.ix_(goodidx, goodidx)]

# dictionary of within vs between receptor similarity
d = dict({'within': autorad_sim[mask][np.where(withbet[mask] == 1)],
          'between': autorad_sim[mask][np.where(withbet[mask] == 0)]})
df_fc = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))

# spin null
emp = np.mean(df_fc['within']) - np.mean(df_fc['between'])
null = np.zeros((nspins, ))
for i in range(nspins):
    rsn_null = rsn_mapping[spins[:, i]]
    withbet_null = np.zeros((nnodes, nnodes))
    for k in range(nnodes):
        for j in range(nnodes):
            if rsn_null[k] == rsn_mapping[j]:
                withbet_null[k, j] = 1
    withbet_null = withbet_null[np.ix_(goodidx, goodidx)]
    fc_null = dict({'within': autorad_sim[mask]
                   [np.where(withbet_null[mask] == 1)],
                   'between': autorad_sim[mask]
                   [np.where(withbet_null[mask] == 0)]})
    null[i] = np.mean(fc_null['within']) - np.mean(fc_null['between'])

pval_fc = (1 + np.sum(np.abs((null - np.mean(null)))
                      >= abs((emp - np.mean(null))))) / (nspins + 1)

                      
"""
plot connectivity resuts
"""

scidx = sc_weighted[np.ix_(goodidx, goodidx)][mask] != 0

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
sns.boxplot(data=df_sc, ax=ax1)
ax1.set_title('p = ' + str(pval_sc)[:5])
sns.boxplot(data=df_fc, ax=ax2)
ax2.set_title('p = ' + str(pval_fc)[:5])
sns.regplot(x=autorad_sim[mask][scidx],
            y=sc_weighted[np.ix_(goodidx, goodidx)][mask][scidx],
            scatter=False, ax=ax3)
ax3.scatter(autorad_sim[mask][scidx],
            sc_weighted[np.ix_(goodidx, goodidx)][mask][scidx], s=5)
r, p = pearsonr(autorad_sim[mask][scidx],
                sc_weighted[np.ix_(goodidx, goodidx)][mask][scidx])
ax3.set_xlabel('receptor similarity')
ax3.set_ylabel('weighted sc')
ax3.set_title('r = ' + str(r)[:4] + ', p = ' + str(p)[:5])
sns.regplot(x=autorad_sim[mask],
            y=fc[np.ix_(goodidx, goodidx)][mask],
            scatter=False, ax=ax4)
ax4.scatter(autorad_sim[mask], fc[np.ix_(goodidx, goodidx)][mask], s=5)
r, p = pearsonr(autorad_sim[mask], fc[np.ix_(goodidx, goodidx)][mask])
ax4.set_xlabel('receptor similarity')
ax4.set_ylabel('functional connectivity')
ax4.set_title('r = ' + str(r)[:4] + ', p = ' + str(p)[:5])
plt.tight_layout()
plt.savefig(path+'figures/schaefer100/autorad_connectivity.eps')


"""
structure-function coupling
"""

co = metrics.communicability_wei(sc_weighted)

rsq_sc = np.zeros([len(autorad_sim), ])
rsq_r = np.zeros([len(autorad_sim), ])
rnull = np.zeros([len(autorad_sim), nspins])

for i in range(len(rsq_sc)):
    print(i)
    y = fc[:, i]
    x1 = co[:, i]
    x2 = autorad_sim[:, i]

    x_sc = zscore(x1[goodidx]).reshape(-1, 1)
    x_r = zscore(np.stack((x1[goodidx], x2), axis=1))
    rsq_sc[i], res_sc = get_reg_r_sq(x_sc, y[goodidx])
    rsq_r[i], res_r = get_reg_r_sq(x_r, y[goodidx])

    for s in range(nspins):
        x2null = x2[perms[:, s]]
        x_rnull = zscore(np.stack((x1[goodidx], x2null), axis=1))
        rnull[i, s], _ = get_reg_r_sq(x_rnull, y[goodidx])

rpvals = np.zeros([len(autorad_sim), ])
for i in range(len(rpvals)):
    rpvals[i] = get_perm_p(rsq_r[i], rnull[i, :])
rpvals = multipletests(rpvals, method='fdr_bh')[1]
np.save(path+'results/rsq_null_autorad.npy', rnull)

plt.ion()
fig, ax = plt.subplots()
plt.scatter(rsq_sc, rsq_r, c=(rpvals < 0.05).astype(int))
plt.plot(rsq_sc, rsq_sc, 'k-', linewidth=.5)
plt.xlabel('Rsq from SC only')
plt.ylabel('Rsq from SC + Receptors')
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.savefig(path+'figures/schaefer100/scatter_scfc_coupling_aut.eps')

brain = plotting.plot_fsaverage(data=add_hem_for_plotting(rsq_r - rsq_sc, nnodes*2),
                                lhannot=annot.lh, rhannot=annot.rh,
                                noplot=schaefer['labels'][badidx],
                                colormap=cmap_seq,
                                vmin=0, vmax=max(rsq_r - rsq_sc),
                                views=['lat', 'med'],
                                data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/schaefer100/surface_scfc_coupling_aut.eps')

"""
dynamics (MEG)
"""

power = np.genfromtxt(path+'data/MEG/power_scale100.csv', delimiter=',')
power_band = ["delta", "theta", "alpha", "beta", "gamma1", "gamma2"]

model_metrics = dict([])
model_pval = np.zeros((len(power_band), ))

for i in range(len(power_band)):
    print(i)
    m, _ = stats.get_dominance_stats(zscore(autorad_schaefer100[goodidx, :]),
                                     zscore(power[goodidx, i]))
    model_metrics[power_band[i]] = m
    # get model pval
    emp, _ = get_reg_r_sq(zscore(autorad_schaefer100[goodidx, :]),
                          zscore(power[goodidx, i]))
    null = np.zeros((nspins, ))
    for s in range(nspins):
        Xnull = autorad_schaefer100[goodidx[perms[:, s]], :]
        null[s], _ = get_reg_r_sq(zscore(Xnull), zscore(power[goodidx, i]))
    model_pval[i] = (1 + sum(null > emp))/(nspins + 1)

dominance = np.zeros((len(power_band), len(receptor_names)))

for i in range(len(model_metrics)):
    tmp = model_metrics[power_band[i]]
    dominance[i, :] = tmp["total_dominance"]
np.save(path+'results/dominance_power_autorad.npy', dominance)

plt.ion()
plt.figure()
plt.bar(np.arange(len(power_band)), np.sum(dominance, axis=1),
        tick_label=power_band)
plt.xticks(rotation='vertical')
plt.ylim([0.5, 0.95])
plt.tight_layout()
plt.savefig(path+'figures/schaefer100/bar_dominance_power_aut.eps')

model_pval = multipletests(model_pval, method='fdr_bh')[1]
dominance[np.where(model_pval >= 0.05)[0], :] = 0

plt.ion()
plt.figure()
sns.heatmap(dominance / np.sum(dominance, axis=1)[:, None],
            xticklabels=receptor_names, yticklabels=power_band,
            cmap=cmap_seq, linewidths=.5)
plt.tight_layout()
plt.savefig(path+'figures/schaefer100/heatmap_dominance_power_aut.eps')

"""
cognition (PLS)
"""

# load neurosynth data
neurosynth = pd.read_csv(path+'data/neurosynth/ns_scale100.csv',
                         delimiter=',')
neurosynth = neurosynth.drop(columns='Unnamed: 0')
X = autorad_schaefer100[goodidx, :]
Y = zscore(neurosynth.iloc[goodidx])

pls_result = pyls.behavioral_pls(X, Y, n_boot=nspins, n_perm=0,
                                 test_split=0, seed=1234)
pyls.save_results(path+'results/pls_result_autorad.hdf5', pls_result)
petpls = pyls.load_results(path+'results/pls_result.hdf5')
lv = 0

# plot scores
brain = plotting.plot_fsaverage(data=add_hem_for_plotting(pls_result['x_scores'][:, lv], nnodes*2),
                                lhannot=annot.lh, rhannot=annot.rh,
                                noplot=schaefer['labels'][badidx],
                                colormap=cmap_div,
                                vmin=-max(abs(pls_result['x_scores'][:, lv])),
                                vmax=max(abs(pls_result['x_scores'][:, lv])),
                                views=['lat', 'med'],
                                data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/schaefer100/surface_autorad_xscores.eps')

brain = plotting.plot_fsaverage(data=add_hem_for_plotting(pls_result['y_scores'][:, lv], nnodes*2),
                                lhannot=annot.lh, rhannot=annot.rh,
                                noplot=schaefer['labels'][badidx],
                                colormap=cmap_div,
                                vmin=-max(abs(pls_result['y_scores'][:, lv])),
                                vmax=max(abs(pls_result['y_scores'][:, lv])),
                                views=['lat', 'med'],
                                data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/schaefer100/surface_autorad_yscores.eps')

# plot receptor loadings
# I'm lazy so just going to flip x and y in pls to get x CI
xload = pyls.behavioral_pls(Y, X, n_boot=10000, n_perm=0, test_split=0)
err = (xload["bootres"]["y_loadings_ci"][:, lv, 1]
      - xload["bootres"]["y_loadings_ci"][:, lv, 0]) / 2
sorted_idx = np.argsort(xload["y_loadings"][:, lv])
plt.figure()
plt.bar(range(len(receptor_names)), xload["y_loadings"][sorted_idx, lv],
        yerr=err[sorted_idx])
plt.xticks(range(len(receptor_names)),
           labels=[receptor_names[i] for i in sorted_idx],
           rotation='vertical')
plt.ylabel("Receptor loadings")
plt.tight_layout()
plt.savefig(path+'figures/schaefer100/bar_pls_rload_autorad.eps')

# plot term loadings
err = (pls_result["bootres"]["y_loadings_ci"][:, lv, 1]
      - pls_result["bootres"]["y_loadings_ci"][:, lv, 0]) / 2
relidx = (abs(pls_result["y_loadings"][:, lv]) - err) > 0  # CI doesnt cross 0
sorted_idx = np.argsort(pls_result["y_loadings"][relidx, lv])
plt.figure(figsize=(10, 5))
plt.bar(np.arange(sum(relidx)), np.sort(pls_result["y_loadings"][relidx, lv]),
        yerr=err[relidx][sorted_idx])
plt.xticks(np.arange(sum(relidx)), labels=neurosynth.columns[relidx][sorted_idx],
           rotation='vertical')
plt.ylabel("Cognitive term loadings")
plt.tight_layout()
plt.savefig(path+'figures/schaefer100/bar_pls_tload_autorad.eps')

# pet vs autorad pls
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
sns.regplot(x=pls_result["x_scores"][:, lv],
            y=petpls["x_scores"][goodidx, lv],
            marker='.', scatter_kws={'s' : 5}, ax=ax1)
r, p = corr_perm(petpls["x_scores"][goodidx, lv],
                 pls_result["x_scores"][:, lv], perms, nspins)
ax1.set_xlabel('receptor scores (autoradiography)')
ax1.set_ylabel('receptor scores (pet)')
ax1.set_title('r = ' + str(r)[:5] + ', p = ' + str(p)[:6])
ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')

sns.regplot(x=pls_result["y_scores"][:, lv],
            y=petpls["y_scores"][goodidx, lv],
            marker='.', scatter_kws={'s' : 5}, ax=ax2)
r, p = corr_perm(petpls["y_scores"][goodidx, lv],
                 pls_result["y_scores"][:, lv], perms, nspins)
ax2.set_xlabel('cognitive scores (autoradiography)')
ax2.set_ylabel('cognitive scores (pet)')
ax2.set_title('r = ' + str(r)[:5] + ', p = ' + str(p)[:6])
ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')

plt.tight_layout()
plt.savefig(path+'figures/schaefer100/scatter_plsscores_pet_autorad.eps')


"""
Disorder profiles (ENIGMA)
"""

# load the enigma cortical thickness cohens d ("abnormality") maps
ct = np.genfromtxt(path+'data/enigma_atrophy.csv', delimiter=',')
disorders = ['22q', 'adhd', 'asd', 'epilepsy_gge', 'epilepsy_rtle',
             'epilepsy_ltle', 'depression', 'ocd', 'schizophrenia',
             'bipolar', 'obesity', 'schizotypy', 'park']

model_metrics = dict([])
model_pval = np.zeros((len(disorders), ))
perms = np.zeros((33, nspins))
for i in range(nspins):
    perms[:, i] = np.random.permutation(33)
perms = perms.astype(int)

for i in range(len(disorders)):
    print(i)
    m, _ = stats.get_dominance_stats(zscore(autorad_cammoun033),
                                     zscore(ct[:33, i]))
    model_metrics[disorders[i]] = m
    # get model pval
    emp, _ = get_reg_r_sq(zscore(autorad_cammoun033),
                          zscore(ct[:33, i]))
    null = np.zeros((nspins, ))
    for s in range(nspins):
        Xnull = autorad_cammoun033[perms[:, s], :]
        null[s], _ = get_reg_r_sq(zscore(Xnull), zscore(ct[:33, i]))
    model_pval[i] = (1 + sum(null > emp))/(nspins + 1)

dominance = np.zeros((len(disorders), len(receptor_names)))

for i in range(len(model_metrics)):
    tmp = model_metrics[disorders[i]]
    dominance[i, :] = tmp["total_dominance"]
np.save(path+'results/dominance_enigma_autorad.npy', dominance)

plt.ion()
plt.figure()
plt.bar(np.arange(len(disorders)), np.sum(dominance, axis=1),
        tick_label=disorders)
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.savefig(path+'figures/schaefer100/bar_dominance_enigma_aut.eps')

model_pval = multipletests(model_pval, method='fdr_bh')[1]
dominance[np.where(model_pval >= 0.05)[0], :] = 0

plt.ion()
plt.figure()
sns.heatmap(dominance / np.sum(dominance, axis=1)[:, None],
            xticklabels=receptor_names, yticklabels=disorders,
            cmap=cmap_seq, linewidths=.5)
plt.tight_layout()
plt.savefig(path+'figures/schaefer100/heatmap_dominance_enigma_aut.eps')
