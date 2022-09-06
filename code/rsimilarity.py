"""
Figure 2: Introduction to the receptor data
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
from netneurotools import datasets, stats, plotting
from scipy.stats import zscore, pearsonr, f_oneway
from scipy.spatial.distance import squareform, pdist
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from nilearn.datasets import fetch_atlas_schaefer_2018
import neuromaps as nmap
from neuromaps import parcellate


def exponential(x, a, b, c):
    return a * np.exp(b * x) + c


def corr_spin(x, y, spins, nspins):
    rho, _ = pearsonr(x, y)
    null = np.zeros((nspins,))

    # null correlation
    for i in range(nspins):
        null[i], _ = pearsonr(x[spins[:, i]], y)

    pval = (1 + sum(abs((null - np.mean(null))) >
                    abs((rho - np.mean(null))))) / (nspins + 1)
    return rho, pval, null


def get_perm_p(emp, null):
    return (1 + sum(abs(null - np.nanmean(null))
                    > abs(emp - np.nanmean(null)))) / (len(null) + 1)


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


"""
set-up
"""

path = 'C:/Users/justi/OneDrive - McGill University/MisicLab/proj_receptors/\
github/hansen_receptors/'

# get the parcellation, coordinates, etc
scale = 'scale100'

schaefer = fetch_atlas_schaefer_2018(n_rois=100)
annot = datasets.fetch_schaefer2018('fsaverage')['100Parcels7Networks']
nnodes = len(schaefer['labels'])
coords = np.genfromtxt(path+'data/schaefer/coordinates/Schaefer_100_centres.txt')[:, 1:]
hemiid = np.zeros((nnodes, ))
hemiid[:int(nnodes/2)] = 1
nspins = 10000
spins = stats.gen_spinsamples(coords, hemiid, n_rotate=nspins, seed=1234)
eu = squareform(pdist(coords, metric='euclidean'))

# load the receptor data
receptor_data = np.genfromtxt(path+'results/receptor_data_'+scale+'.csv', delimiter=',')
receptor_names = np.load(path+'data/receptor_names_pet.npy')
receptor_similarity = np.corrcoef(zscore(receptor_data))
mask = np.triu(np.ones(nnodes), 1) > 0

# colourmaps
cmap = np.genfromtxt(path+'data/colourmap.csv', delimiter=',')
cmap_div = ListedColormap(cmap)
cmap_seq = ListedColormap(cmap[128:, :])

# rsn networks for plotting
rsn_mapping = []
for row in range(len(schaefer['labels'])):
    rsn_mapping.append(schaefer['labels'][row].decode('utf-8').split('_')[2])
rsn_mapping = np.array(rsn_mapping)

"""
plot
"""

plt.ion()

# receptor similarity
inds = plotting.sort_communities(receptor_similarity, rsn_mapping)
bounds = plotting._grid_communities(rsn_mapping)
bounds[0] += 0.2
bounds[-1] -= 0.2

fig, ax = plt.subplots()
sns.heatmap(data=receptor_similarity[np.ix_(inds, inds)], cmap=cmap_div,
            vmin=-1, vmax=1, ax=ax, cbar=False, square=True,
            xticklabels=False, yticklabels=False)
for n, edge in enumerate(np.diff(bounds)):
    ax.add_patch(patches.Rectangle((bounds[n], bounds[n]),
                                    edge, edge, fill=False, linewidth=2,
                                    edgecolor='black'))
plt.tight_layout()
plt.savefig(path+'figures/schaefer100/heatmap_receptor_similarity.eps')

# histogram
plt.figure()
ax = sns.distplot(receptor_similarity[mask])
ax.set(xlabel = 'receptor similarity')
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.savefig(path+'figures/schaefer100/hist_receptor_similarity.eps')

# euclidean distance
p0 = [1, -0.05, -0.1]  # initial parameter guesses
pars, _ = curve_fit(exponential, eu[mask], receptor_similarity[mask], p0=p0)

linresid = get_reg_r_sq(eu[mask].reshape(-1, 1), receptor_similarity[mask])[1]
expfit = exponential(eu[mask], pars[0], pars[1], pars[2])
expresid = sum((receptor_similarity[mask] - expfit)**2)

fig, ax = plt.subplots()
ax.scatter(eu[mask], receptor_similarity[mask], s=5)
ax.plot(np.arange(10, 160), exponential(np.arange(10, 160),
                                         pars[0], pars[1], pars[2]), color='red')
ax.set_xlabel('euclidean distance')
ax.set_ylabel('receptor similarity')
ax.set_title('y = ' + str(pars[0])[:4]
          + ' * exp(' + str(pars[1])[:5]
          + ' * x) + '+ str(pars[2])[:5])
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.tight_layout()
plt.savefig(path+'figures/schaefer100/scatter_distance.eps')

# PC1
pca = PCA(n_components=1)
pc1 = np.squeeze(pca.fit_transform(zscore(receptor_data)))

brain = plotting.plot_fsaverage(data=pc1,
                                lhannot=annot.lh, rhannot=annot.rh,
                                colormap=cmap_div,
                                vmin=-np.max(np.abs(pc1)),
                                vmax=np.max(np.abs(pc1)),
                                views=['lat', 'med'],
                                data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/schaefer100/surface_pc1.eps')

# PC1 within mesulam classes
me = np.genfromtxt(path+'data/mesulam_scale100.csv', delimiter=',')
me_names = ['plmb', 'het', 'uni', 'idio']
f, p = f_oneway(pc1[me==1], pc1[me==2], pc1[me==3], pc1[me==4])

fig, ax = plt.subplots()
sns.violinplot(x=me, y=pc1, inner=None, color=".8", ax=ax)
sns.stripplot(x=me, y=pc1, ax=ax)
ax.set_xticklabels(me_names)
ax.set_ylabel('PC1sim')
plt.tight_layout()
plt.savefig(path+'figures/schaefer100/stripplot_mes_classes_pc1.eps')

# PC1 and synapse density
ucbj_surface = nmap.datasets.fetch_annotation(desc='ucbj')
parc = parcellate.Parcellater(schaefer['maps'], 'MNI152')
ucbj = zscore(np.squeeze(parc.fit_transform(ucbj_surface, 'MNI152').T))
r, p, _ = corr_spin(pc1, ucbj, spins, nspins)
plt.figure()
sns.regplot(pc1, ucbj, scatter=False)
plt.scatter(pc1, ucbj)
plt.xlabel('PC1')
plt.ylabel('synapse density')
plt.title(['rho = ' + str(r)[:4] + ', pspin = ' + str(p)[:5]])
plt.tight_layout()
plt.savefig(path+'figures/schaefer100/scatter_pc1_ucbj.eps')

# correlating receptors, 4 ways

exc = ['5HT2a', '5HT4', '5HT6', 'D1', 'mGluR5', 'A4B2', 'M1', 'NMDA']
inh = ['5HT1a', '5HT1b', 'CB1', 'D2', 'GABAa', 'H3', 'MOR']
mami = ['5HT1a', '5HT1b', '5HT2a', '5HT4', '5HT6', '5HTT', 'D1', 'D2', 'DAT', 'H3', 'NET']
nmami = list(set(receptor_names) - set(mami))
metab = ['5HT1a', '5HT1b', '5HT2a', '5HT4', '5HT6', 'CB1', 'D1', 'D2', 'H3', 'M1', 'mGluR5', 'MOR']
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

classes = np.zeros((len(receptor_names), 4))
classes[i_exc, 0] = 1
classes[i_inh, 0] = 2
classes[i_mami, 1] = 1
classes[i_nmami, 1] = 2
classes[i_metab, 2] = 1
classes[i_iono, 2] = 2
classes[i_gs, 3] = 1
classes[i_gi, 3] = 2
classes[i_gq, 3] = 3

class_names = ['exc/inh', 'monoamine', 'ionotropic/metabotropic', 'Gs/Gi/Gq']

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs = axs.ravel()
for i in range(len(axs)):
    inds = plotting.sort_communities(np.corrcoef(zscore(receptor_data).T), classes[:, i])
    bounds = plotting._grid_communities(classes[:, i])
    sns.heatmap(data=np.corrcoef(zscore(receptor_data).T)[np.ix_(inds, inds)],
                vmin=-1, vmax=1, ax=axs[i], cbar=False, square=True, cmap=cmap_div,
                linewidths=.5, xticklabels=receptor_names[inds], yticklabels=receptor_names[inds])
    for n, edge in enumerate(np.diff(bounds)):
        axs[i].add_patch(patches.Rectangle((bounds[n], bounds[n]), 
                                            edge, edge, fill=False,
                                            linewidth=1, edgecolor='black'))
    axs[i].set_title(class_names[i])
plt.tight_layout()
plt.savefig(path+'figures/schaefer100/heatmap_receptor_corr_byclass.eps')
