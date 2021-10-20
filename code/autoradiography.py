# -*- coding: utf-8 -*-
"""
Autoradiography 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from netneurotools import datasets, utils, stats, plotting
from scipy.stats import zscore, pearsonr
from scipy.linalg import expm, eig
from scipy.optimize import curve_fit
from matplotlib.colors import ListedColormap
from bct import distance
from scipy.spatial.distance import squareform, pdist
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_random_state


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


def get_f_stat(rss1, rss2, p1, p2, n):
    return ((rss1 - rss2) / (p2 - p1)) / (rss2/(n - p2))


def brodmann2dk(M, duplicate, mapping):
    """
    Converts 44 autoradiography regions to 33 Desikan Killiany.

    Parameters
    ----------
    M : (44, 15) np array
        Autoradiography receptor densities from Zilles & Palomero-Gallagher.
    duplicate : (1, 8) array
        Autoradiography regions to duplicate.
    mapping : (1, 52) np array
        DK indices mapped to Brodmann/Jubrain regions including duplicates.

    Returns
    -------
    convertedM : (33, 15) np array
        Autoradiography densities mapped to 33 DK regions (insula excluded).

    """

    rep = np.ones((M.shape[0], ), dtype=int)  # duplicate regions
    rep[duplicate] = 2
    M = np.repeat(M, rep, 0)

    # convert to dk
    n_dknodes = max(mapping) + 1  # number of nodes in dk atlas (left hem only)

    u = np.unique(mapping)
    convertedM = np.zeros((n_dknodes, M.shape[1]))
    for i in range(len(u)):
        if sum(mapping == u[i]) > 1:
            convertedM[u[i], :] = np.mean(M[mapping == u[i], :], axis=0)
        else:
            convertedM[u[i], :] = M[mapping == u[i], :]

    return convertedM


def corr_spin(x, y, spins, nspins):
    rho, _ = pearsonr(x, y)
    null = np.zeros((nspins,))

    if len(x) == spins.shape[0] - 1:  # if insula is missing
        np.append(x, np.nan)
        np.append(y, np.nan)

    # convert to dataframe for better handling of nans
    # in the case that insula is missing
    df = pd.DataFrame(np.concatenate(x, y), columns=['x', 'y'])

    # null correlation
    for i in range(nspins):
        null[i] = df["x"].corr(df["y"][spins[:, i]])

    pval = (1 + sum(abs((null - np.mean(null))) >
                    abs((rho - np.mean(null))))) / (nspins + 1)
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
    newdata[int(nnodes_full/2):int(nnodes_full/2)+len(data)] = data
    return newdata



"""
set-up
"""

path = 'C:/Users/justi/OneDrive - McGill University/MisicLab/proj_receptors/\
github/hansen_receptors/'

# load PET data
PET_data = np.genfromtxt(path+'results/receptor_data_scale033.csv', delimiter=',')

# load spins
cammoun = datasets.fetch_cammoun2012()
info = pd.read_csv(cammoun['info'])
leftcortex = info.query('scale == "scale033" \
                    & structure == "cortex" \
                    & hemisphere == "L"')['id']
leftcortex = np.array(leftcortex) - 1  # python indexing
cortex = info.query('scale == "scale033" & structure == "cortex"')['id']
cortex = np.array(cortex) - 1  # python indexing
coords = utils.get_centroids(cammoun['scale033'], image_space=True)
coords = coords[leftcortex, :]
nspins = 10000
spins = stats.gen_spinsamples(coords, hemiid=np.ones((len(leftcortex),)),
                              n_rotate=nspins, seed=1234)
eu = squareform(pdist(utils.get_centroids(cammoun['scale033'], image_space=True),
                      metric="euclidean"))

idx = np.arange(34, 67, 1)  # index of left cortex without insula
nnodes = len(idx)

# load sc fc
sc = np.load(path+'data/lausanne/sc_binary.npy')
sc_left = sc[idx[:, None], idx[None, :]]  # no insula
sc_weighted = np.load(path+'data/lausanne/sc_weighted.npy')
fc = np.load(path+'data/lausanne/fc_weighted.npy')
fc = fc[idx[:, None], idx[None, :]]

# colourmaps
cmap = np.genfromtxt(path+'/data/colourmap.csv', delimiter=',')
cmap_div = ListedColormap(cmap)
cmap_seq = ListedColormap(cmap[128:, :])

"""
make autoradiography receptor matrix
"""

receptdata_s = np.load(path+'data/autoradiography/ReceptData_S.npy')  # supra
receptdata_g = np.load(path+'data/autoradiography/ReceptData_G.npy')  # granular
receptdata_i = np.load(path+'data/autoradiography/ReceptData_I.npy')  # infra
receptor_names = np.load(path+'data/autoradiography/ReceptorNames.npy')

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

# convert
# note: insula (last idx of cammoun atlas) is missing
receptdata_s = zscore(brodmann2dk(receptdata_s, duplicate, mapping))
receptdata_g = zscore(brodmann2dk(receptdata_g, duplicate, mapping))
receptdata_i = zscore(brodmann2dk(receptdata_i, duplicate, mapping))

# average across layers
# final region x receptor autoradiography receptor dataset
autorad_data = np.mean(np.stack((receptdata_s,
                                 receptdata_g,
                                 receptdata_i), axis=2), axis=2)
np.save(path+'results/autorad_data.npy', autorad_data)

"""
data visualization
"""

# mean density
annot = datasets.fetch_cammoun2012('fsaverage')['scale033']
brain = plotting.plot_fsaverage(data=add_hem_for_plotting(np.mean(autorad_data, axis=1), len(cortex)),
                                lhannot=annot.lh, rhannot=annot.rh,
                                noplot=['insula'], order='rl',
                                colormap=cmap_div,
                                vmin=-max(abs(np.mean(autorad_data, axis=1))),
                                vmax=max(abs(np.mean(autorad_data, axis=1))),
                                views=['lat', 'med'],
                                data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/surface_autorad_recept_density.eps')

# correlating receptors
plt.ion()
plt.figure()
sns.heatmap(np.corrcoef(autorad_data.T), vmin=-1, vmax=1, cmap=cmap_div,
            cbar=False, square=True, linewidths=.5,
            xticklabels=receptor_names, yticklabels=receptor_names)
plt.tight_layout()
plt.savefig(path+'figures/heatmap_autorad_recept_corr.eps')

# correlating regions
plt.ion()
plt.figure()
sns.heatmap(np.corrcoef(autorad_data), vmin=-1, vmax=1, cmap=cmap_div,
            cbar=False, square=True, linewidths=.5,
            xticklabels=False, yticklabels=False)
plt.tight_layout()
plt.savefig(path+'figures/heatmap_autorad_region_corr.eps')

"""
Compare with PET receptor similarity
"""
PET_sim = np.corrcoef(zscore(PET_data)[idx, :])  # left hem + no insula
autorad_sim = np.corrcoef(autorad_data)
mask = np.triu(np.ones(autorad_data.shape[0]), 1) > 0
r, p = pearsonr(autorad_sim[mask], PET_sim[mask])

plt.ion()
plt.figure()
sns.regplot(autorad_sim[mask], PET_sim[mask], scatter=False)
plt.scatter(autorad_sim[mask], PET_sim[mask], s=5)
plt.xlabel('autoradiography receptor similarity')
plt.ylabel('PET receptor similarity')
plt.title('rho = ' + str(r)[:5] + ', p = ' + str(p)[:5])
plt.savefig(path+'figures/scatter_autorad_v_pet.eps')

"""
structure: connected vs not connected
"""

# set up dictionary with connected vs not connected receptor similarity
d = dict({'connected': autorad_sim[mask][np.where(sc_left[mask] == 1)],
          'not connected': autorad_sim[mask][np.where(sc_left[mask] == 0)]})
df_sc = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))

emp = np.mean(df_sc['connected']) - np.mean(df_sc['not connected'])

null = np.zeros((nspins, ))
for i in range(nspins):
    sc_rewired, _ = match_length_degree_distribution(sc,
                                                     eu[cortex[None, :], cortex[:, None]],
                                                     10, len(cortex)*20)
    sc_rew_left = sc[idx[:, None], idx[None, :]]
    null[i] = np.mean(autorad_sim[mask]
                      [np.where(sc_rew_left[mask] == 1)]) \
              - np.mean(autorad_sim[mask]
                        [np.where(sc_rew_left[mask] == 0)])
np.save(path+'results/sc_edge-degree-preserving_null_autorad.npy', null)

pval_sc = (1 + np.sum(np.abs((null - np.mean(null)))
                      >= abs((emp - np.mean(null))))) / (nspins + 1)

"""
function: within vs between
"""

rsn_mapping = np.array(info.query('scale == "scale033"')['yeo_7'])

withbet = np.zeros((len(leftcortex), len(leftcortex)))
for k in range(nnodes):
    for j in range(nnodes):
        if rsn_mapping[leftcortex[k]] == rsn_mapping[leftcortex[j]]:
            withbet[k, j] = 1
withbet = withbet[:-1, :-1]

# dictionary of within vs between receptor similarity
d = dict({'within': autorad_sim[mask][np.where(withbet[mask] == 1)],
          'between': autorad_sim[mask][np.where(withbet[mask] == 0)]})
df_fc = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))

# spin null
emp = np.mean(df_fc['within']) - np.mean(df_fc['between'])
null = np.zeros((nspins, ))
for i in range(nspins):
    rsn_null = rsn_mapping[leftcortex[spins[:, i]]]
    withbet_null = np.zeros((len(leftcortex), len(leftcortex)))
    for k in range(nnodes):
        for j in range(nnodes):
            if rsn_null[k] == rsn_mapping[leftcortex][j]:
                withbet_null[k, j] = 1
    withbet_null = withbet_null[:-1, :-1]
    fc_null = dict({'within': autorad_sim[mask]
                   [np.where(withbet_null[mask] == 1)],
                   'between': autorad_sim[mask]
                   [np.where(withbet_null[mask] == 0)]})
    null[i] = np.mean(fc_null['within']) - np.mean(fc_null['between'])

pval_fc = (1 + np.sum(np.abs((null - np.mean(null)))
                      >= abs((emp - np.mean(null))))) / (nspins + 1)


"""
exponential relationship
"""

p0 = [1, -0.05, -0.1]  # initial parameter guesses
pars, _ = curve_fit(exponential,
                    eu[leftcortex[:, None], leftcortex[None, :]][:-1, :-1][mask],
                    autorad_sim[mask], p0=p0)

                      
"""
plot
"""

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
sns.boxplot(data=df_sc, ax=ax1)
sns.boxplot(data=df_fc, ax=ax2)
sns.regplot(x=autorad_sim[mask], y=fc[mask], scatter=False, ax=ax3)
r, p = pearsonr(autorad_sim[mask], fc[mask])
ax3.scatter(autorad_sim[mask], fc[mask], s=5)
ax1.set_title('p = ' + str(pval_sc)[:5])
ax2.set_title('p = ' + str(pval_fc)[:5])
ax3.set_xlabel('functional connectivity')
ax3.set_ylabel('receptor similarity')
ax3.set_title('r = ' + str(r)[:4] + ', pspin = ' + str(p)[:5])
ax4.scatter(eu[leftcortex[:, None], leftcortex[None, :]][:-1, :-1][mask],
            autorad_sim[mask], s=5)
ax4.plot(np.arange(10, 160), exponential(np.arange(10, 160),
                                         pars[0], pars[1], pars[2]), color='red')
ax4.set_xlabel('euclidean distance')
ax4.set_ylabel('receptor similarity')
ax4.set_title('y = ' + str(pars[0])[:4]
              + ' * exp(' + str(pars[1])[:5]
              + ' * x) + '+ str(pars[2])[:5])
plt.tight_layout()
plt.savefig(path+'figures/autorad_connectivity.eps')


"""
structure-function coupling
"""

sp = distance.distance_bin(sc)  # path length
sc_norm = sc / max(eig(sc)[0]).real  # normalize by largest eigenvalue
co = expm(sc_norm)  # communicability

rsq_sc = np.zeros([nnodes, ])
rsq_r = np.zeros([nnodes, ])
ftest = np.zeros([nnodes, ])

for i in range(nnodes):
    y = fc[:, i]
    x1 = sp[idx, i]
    x2 = co[idx, i]
    x3 = eu[cortex[idx, None], cortex[None, idx]][:, i]
    x4 = autorad_sim[:, i]

    x_sc = zscore(np.stack((x1, x2, x3), axis=1))
    x_r = zscore(np.stack((x1, x2, x3, x4), axis=1))
    rsq_sc[i], res_sc = get_reg_r_sq(x_sc, y)
    rsq_r[i], res_r = get_reg_r_sq(x_r, y)

    ftest[i] = get_f_stat(res_sc, res_r, x_sc.shape[1], x_r.shape[1], nnodes-1)

fcrit=4.18

plt.ion()
fig, ax = plt.subplots()
plt.scatter(rsq_sc, rsq_r, c=(ftest>fcrit).astype(int))
plt.plot(rsq_sc, rsq_sc, 'k-')
plt.xlabel('Rsq from SC only')
plt.ylabel('Rsq from SC + Receptors')
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.savefig(path+'figures/scatter_autorad_scfc_coupling_multi.eps')

brain = plotting.plot_fsaverage(data=add_hem_for_plotting(rsq_r - rsq_sc, len(cortex)),
                                order='rl',
                                lhannot=annot.lh, rhannot=annot.rh,
                                noplot=['insula'],
                                colormap=cmap_div,
                                vmin=-max(abs(rsq_r - rsq_sc)),
                                vmax=max(abs(rsq_r - rsq_sc)),
                                views=['lat', 'med'],
                                data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/surface_autorad_scfc_coupling_multi.eps')
