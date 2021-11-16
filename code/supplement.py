# -*- coding: utf-8 -*-
"""
Supplemental analyses
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from netneurotools import datasets, plotting
from scipy.stats import zscore, pearsonr
from sklearn.linear_model import LinearRegression


def regress_age(age, receptor):
    lin_reg = LinearRegression()
    lin_reg.fit(age, receptor)
    yhat = lin_reg.predict(age)
    resid = receptor - yhat
    return resid


path = 'C:/Users/justi/OneDrive - McGill University/MisicLab/proj_receptors/\
github/hansen_receptors/'

scale = 'scale033'  # or scale060 or scale125

cammoun = datasets.fetch_cammoun2012()
info = pd.read_csv(cammoun['info'])
cortex = info.query('scale == "scale033" & structure == "cortex"')['id']
cortex = np.array(cortex) - 1  # python indexing
nnodes = len(cortex)

# colourmaps
cmap = np.genfromtxt(path+'data/colourmap.csv', delimiter=',')
cmap_div = ListedColormap(cmap)
cmap_seq = ListedColormap(cmap[128:, :])

"""
compare different tracers
"""

receptors_csv = [path+'data/PET_parcellated/'+scale+'/5HT1a_way_hc36_savli.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT1a_cumi_hc8_beliveau.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT1b_az_hc36_beliveau.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT1b_p943_hc22_savli.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT1b_p943_hc65_gallezot.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT2a_cimbi_hc29_beliveau.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT2a_ALT_hc19_savli.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT2a_mdl_hc3_talbot.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT4_sb20_hc59_beliveau.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT6_gsk_hc30_radhakrishnan.csv',
                 path+'data/PET_parcellated/'+scale+'/5HTT_dasb_hc100_beliveau.csv',
                 path+'data/PET_parcellated/'+scale+'/5HTT_dasb_hc30_savli.csv',
                 path+'data/PET_parcellated/'+scale+'/A4B2_flubatine_hc30_hillmer.csv',
                 path+'data/PET_parcellated/'+scale+'/CB1_FMPEPd2_hc22_laurikainen.csv',
                 path+'data/PET_parcellated/'+scale+'/CB1_omar_hc77_normandin.csv',
                 path+'data/PET_parcellated/'+scale+'/D1_SCH23390_hc13_kaller.csv',
                 path+'data/PET_parcellated/'+scale+'/D2_flb457_hc37_smith.csv',
                 path+'data/PET_parcellated/'+scale+'/D2_flb457_hc55_sandiego.csv',
                 path+'data/PET_parcellated/'+scale+'/D2_fallypride_hc49_jaworska.csv',
                 path+'data/PET_parcellated/'+scale+'/D2_raclopride_hc7_alakurtti.csv',
                 path+'data/PET_parcellated/'+scale+'/DAT_fepe2i_hc6_sasaki.csv',
                 path+'data/PET_parcellated/'+scale+'/DAT_fpcit_hc174_dukart_spect.csv',
                 path+'data/PET_parcellated/'+scale+'/GABAa-bz_flumazenil_hc16_norgaard.csv',
                 path+'data/PET_parcellated/'+scale+'/GABAa_flumazenil_hc6_dukart.csv',
                 path+'data/PET_parcellated/'+scale+'/H3_cban_hc8_gallezot.csv', 
                 path+'data/PET_parcellated/'+scale+'/M1_lsn_hc24_naganawa.csv',
                 path+'data/PET_parcellated/'+scale+'/mGluR5_abp_hc22_rosaneto.csv',
                 path+'data/PET_parcellated/'+scale+'/mGluR5_abp_hc28_dubois.csv',
                 path+'data/PET_parcellated/'+scale+'/mGluR5_abp_hc73_smart.csv',
                 path+'data/PET_parcellated/'+scale+'/MU_carfentanil_hc204_kantonen.csv',
                 path+'data/PET_parcellated/'+scale+'/MU_carfentanil_hc39_turtonen.csv',
                 path+'data/PET_parcellated/'+scale+'/NAT_MRB_hc10_hesse.csv',
                 path+'data/PET_parcellated/'+scale+'/NAT_MRB_hc77_ding.csv',
                 path+'data/PET_parcellated/'+scale+'/VAChT_feobv_hc3_spreng.csv',
                 path+'data/PET_parcellated/'+scale+'/VAChT_feobv_hc4_tuominen.csv',
                 path+'data/PET_parcellated/'+scale+'/VAChT_feobv_hc5_bedard_sum.csv',
                 path+'data/PET_parcellated/'+scale+'/VAChT_feobv_hc18_aghourian_sum.csv']

receptors_all = {}

for receptor in receptors_csv:
    name = receptor.split('/')[-1]  # get file name
    name = name.split('.')[0]  # remove .csv
    receptors_all[name] = zscore(np.genfromtxt(receptor, delimiter=',')[cortex])

receptor_data = np.genfromtxt(path+'results/receptor_data_'+scale+'.csv', delimiter=',')
receptor_data = zscore(receptor_data)
receptor_names = np.load(path+'data/receptor_names_pet.npy')

plt.ion()
fig, axs = plt.subplots(3, 5, figsize=(18, 10))
axs = axs.ravel()

# combining the same tracer into one mean map

sns.regplot(x=receptor_data[:, receptor_names.index("5HT1b")],
            y=receptors_all['5HT1b_p943_hc22_savli'], ci=None,
            ax=axs[0])
sns.regplot(x=receptor_data[:, receptor_names.index("5HT1b")],
            y=receptors_all['5HT1b_p943_hc65_gallezot'], ci=None,
            ax=axs[0])
axs[0].set(xlabel="mean map", ylabel="P943")
axs[0].set_title("5HT1b")
axs[0].legend(("Savli", "Gallezot"))

sns.regplot(x=receptor_data[:, receptor_names.index("D2")],
            y=receptors_all['D2_flb457_hc37_sandiego'], ci=None,
            ax=axs[1])
sns.regplot(x=receptor_data[:, receptor_names.index("D2")],
            y=receptors_all['D2_flb457_hc55_sandiego'], ci=None,
            ax=axs[1])
axs[1].set(xlabel="mean map", ylabel="FLB457")
axs[1].set_title("D2")
axs[1].legend(("Sandiego37", "Sandiego55"))

sns.regplot(x=receptor_data[:, receptor_names.index("mGluR5")],
            y=receptors_all['mGluR5_abp_hc22_stijn_lls'], ci=None,
            ax=axs[2])
sns.regplot(x=receptor_data[:, receptor_names.index("mGluR5")],
            y=receptors_all['mGluR5_abp_hc28_stijn_lls'], ci=None,
            ax=axs[2])
sns.regplot(x=receptor_data[:, receptor_names.index("mGluR5")],
            y=receptors_all['mGluR5_abp_hc73_smart'], ci=None,
            ax=axs[2])
axs[2].set(xlabel="mean map", ylabel="ABP688")
axs[2].set_title("mGluR5")
axs[2].legend(("Servaes", "Dubois", "Smart"))

sns.regplot(receptor_data[:, receptor_names.index("VAChT")],
            receptors_all['VAChT_feobv_hc3_spreng'], ci=None,
            ax=axs[3])
sns.regplot(receptor_data[:, receptor_names.index("VAChT")],
            receptors_all['VAChT_feobv_hc4_tuominen'], ci=None,
            ax=axs[3])
sns.regplot(receptor_data[:, receptor_names.index("VAChT")],
            receptors_all['VAChT_feobv_hc5_bedard_sum'], ci=None,
            ax=axs[3])
sns.regplot(receptor_data[:, receptor_names.index("VAChT")],
            receptors_all['VAChT_feobv_hc18_aghourian_sum'], ci=None,
            ax=axs[3])                     
axs[3].set(xlabel="mean map", ylabel="FEOBV")
axs[3].set_title("VAChT")
axs[3].legend(("Spreng", "Tuominen", "Bedard", "Aghourian"))

# comparing different tracers

t = ["5HT1a", "5HT1b", "5HT2a", "5HTT", "CB1", "D2", "DAT", "GABAa", "MU", "NAT"]
othermap = ["5HT1a_cumi_hc8_beliveau", "5HT1b_az_hc36_beliveau", 
            "5HT2a_ALT_hc19_savli", "5HTT_dasb_hc30_savli",
            "CB1_FMPEPd2_hc22_laurikainen", "D2_fallypride_hc49_jaworska",
            "DAT_fepe2i_hc6_sasaki", "GABAa_flumazenil_hc6_dukart",
            "MU_carfentanil_hc39_turtonen", "NAT_MRB_hc10_hesse"]
for i in range(5, 16):
    sns.regplot(receptor_data[:, receptor_names.index(t[i-5])],
                receptors_all[othermap[i-5]], ci=None, ax=axs[i])
    axs[i].set(xlabel="selected map", ylabel="alternative map")
    axs[i].set_title(t[i-5])

sns.regplot(receptor_data[:, receptor_names.index("5HT2a")],
            receptors_all['5HT2a_mdl_hc3_talbot'], ci=None,
            ax=axs[7])
axs[7].legend(("alt", "MDL"))

sns.regplot(receptor_data[:, receptor_names.index("D2")],
            receptors_all["D2_raclopride_hc7_alakurtti"], ci=None,
            ax=axs[10])
axs[10].legend(("fallypride", "raclopride"))

plt.tight_layout()
plt.savefig(path+'figures/scatter_supplement_tracers.eps')


"""
compare different parcellation resoultions
"""

recept033 = receptor_data
recept060 = np.genfromtxt(path+'results/receptor_data_scale060.csv', delimiter=',')
recept125 = np.genfromtxt(path+'results/receptor_data_scale125.csv', delimiter=',')

# mean density: scale033
annot = datasets.fetch_cammoun2012('fsaverage')['scale033']
brain = plotting.plot_fsaverage(data=np.mean(zscore(recept033), axis=1),
                                lhannot=annot.lh, rhannot=annot.rh,
                                order = 'rl',
                                colormap=cmap_div,
                                vmin=-np.max(np.abs(np.mean(zscore(recept033), axis=1))),
                                vmax=np.max(np.abs(np.mean(zscore(recept033), axis=1))),
                                views=['lat', 'med'],
                                data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/surface_recept_density.eps')

# mean density: scale060
annot = datasets.fetch_cammoun2012('fsaverage')['scale060']
brain = plotting.plot_fsaverage(data=np.mean(zscore(recept060), axis=1),
                                lhannot=annot.lh, rhannot=annot.rh,
                                order = 'rl',
                                colormap=cmap_div,
                                vmin=-np.max(np.abs(np.mean(zscore(recept060), axis=1))),
                                vmax=np.max(np.abs(np.mean(zscore(recept060), axis=1))),
                                views=['lat', 'med'],
                                data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/surface_recept_density_scale060.eps')

# mean density: scale125
annot = datasets.fetch_cammoun2012('fsaverage')['scale125']
brain = plotting.plot_fsaverage(data=np.mean(zscore(recept125), axis=1),
                                lhannot=annot.lh, rhannot=annot.rh,
                                order = 'rl',
                                colormap=cmap_div,
                                vmin=-np.max(np.abs(np.mean(zscore(recept125), axis=1))),
                                vmax=np.max(np.abs(np.mean(zscore(recept125), axis=1))),
                                views=['lat', 'med'],
                                data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/surface_recept_density_scale125.eps')

# receptor similarity
plt.ion()
fig, axs = plt.subplots(1, 3)
axs = axs.ravel()
sns.heatmap(np.corrcoef(zscore(recept033)), vmin=-1, vmax=1, cmap=cmap_div,
            cbar=False, square=True, xticklabels=False, yticklabels=False, ax=axs[0])
sns.heatmap(np.corrcoef(zscore(recept060)), vmin=-1, vmax=1, cmap=cmap_div,
            cbar=False, square=True, xticklabels=False, yticklabels=False, ax=axs[1])
sns.heatmap(np.corrcoef(zscore(recept125)), vmin=-1, vmax=1, cmap=cmap_div,
            cbar=False, square=True, xticklabels=False, yticklabels=False, ax=axs[2])
plt.tight_layout()
plt.savefig(path+'figures/heatmap_similarity_mats.eps')


"""
leave-one-out receptor similarity
"""

loo_rho = np.zeros((len(receptor_names), ))
mask = np.triu(np.ones(nnodes), 1) > 0
for k in range(len(receptor_names)):
    # receptor similarity when you remove one disorder
    tmp = np.corrcoef(np.delete(zscore(receptor_data), k, axis=1))
    # correlate with complete receptor similarity matrix
    loo_rho[k], _ = pearsonr(tmp[mask], np.corrcoef(zscore(receptor_data))[mask])

plt.ion()
plt.figure()
ax = sns.violinplot(data=loo_rho)
ax.set(ylabel='correlation')
plt.savefig(path+'figures/violin_loo.eps')


"""
age effects?
"""
# at each brain region, correlate age vector to density vector
age = np.array((26.3, 32.4, 22.6, 25.9, 36.6, 25.1, 33.5, 30.0, 33.0,
                38.8, 61.0, 26.6, 33.4, 31.7, 40.5, 31.5, 32.3, 63.6))
age_corr = np.zeros((nnodes, ))
for k in range(nnodes):
    age_corr[k], _ = pearsonr(age, receptor_data[k, :])

annot = datasets.fetch_cammoun2012('fsaverage')['scale033']
brain = plotting.plot_fsaverage(data=age_corr,
                                lhannot=annot.lh, rhannot=annot.rh,
                                order = 'rl', colormap=cmap_div,
                                vmin=-1, vmax=1, views=['lat', 'med'],
                                data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/surface_age_effect.png')

# regress age out
receptor_data_reg = np.zeros(receptor_data.shape)
for i in range(nnodes):
    receptor_data_reg[i, :] = np.squeeze(regress_age(age.reshape(-1, 1),
                                                     receptor_data[i, :].reshape(-1, 1)))

# plot
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(receptor_data.flatten(), receptor_data_reg.flatten(), s=5)
ax1.set_xlabel('original receptor densities')
ax1.set_ylabel('age regressed receptor densities')
ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
rsim = np.corrcoef(receptor_data)
rsim_reg = np.corrcoef(receptor_data_reg)
mask =  np.triu(np.ones(nnodes), 1) > 0
ax2. scatter(rsim[mask], rsim_reg[mask], s=5)
ax2.set_xlabel('original receptor similarity')
ax2.set_ylabel('age regressed receptor similarity')
ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')
plt.tight_layout()
plt.savefig(path+'figures/age_effects.eps')