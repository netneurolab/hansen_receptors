# -*- coding: utf-8 -*-
"""
Concatenate parcellated PET images into region x receptor matrix of densities.
"""

import numpy as np
import pandas as pd
from netneurotools import datasets, plotting
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


scale = 'scale033'  # or scale060 or scale125

cammoun = datasets.fetch_cammoun2012()
info = pd.read_csv(cammoun['info'])
cortex = info.query('scale == @scale & structure == "cortex"')['id']
cortex = np.array(cortex) - 1  # python indexing
nnodes = len(cortex)

path = 'C:/Users/justi/OneDrive - McGill University/MisicLab/proj_receptors/\
github/hansen_receptors/'

# concatenate the receptors

receptors_csv = [path+'data/PET_parcellated/'+scale+'/5HT1a_way_hc36_savli.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT1b_p943_hc22_savli.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT1b_p943_hc65_gallezot.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT2a_cimbi_hc29_beliveau.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT4_sb20_hc59_beliveau.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT6_gsk_hc30_radnakrishnan.csv',
                 path+'data/PET_parcellated/'+scale+'/5HTT_dasb_hc100_beliveau.csv',
                 path+'data/PET_parcellated/'+scale+'/A4B2_flubatine_hc30_hillmer.csv',
                 path+'data/PET_parcellated/'+scale+'/CB1_omar_hc77_normandin.csv',
                 path+'data/PET_parcellated/'+scale+'/D1_SCH23390_hc13_kaller.csv',
                 path+'data/PET_parcellated/'+scale+'/D2_flb457_hc37_smith.csv',
                 path+'data/PET_parcellated/'+scale+'/D2_flb457_hc55_sandiego.csv',
                 path+'data/PET_parcellated/'+scale+'/DAT_fpcit_hc174_dukart_spect.csv',
                 path+'data/PET_parcellated/'+scale+'/GABAa-bz_flumazenil_hc16_norgaard.csv',
                 path+'data/PET_parcellated/'+scale+'/H3_cban_hc8_gallezot.csv', 
                 path+'data/PET_parcellated/'+scale+'/M1_lsn_hc24_naganawa.csv',
                 path+'data/PET_parcellated/'+scale+'/mGluR5_abp_hc22_rosaneto.csv',
                 path+'data/PET_parcellated/'+scale+'/mGluR5_abp_hc28_dubois.csv',
                 path+'data/PET_parcellated/'+scale+'/mGluR5_abp_hc73_smart.csv',
                 path+'data/PET_parcellated/'+scale+'/MU_carfentanil_hc204_kantonen.csv',
                 path+'data/PET_parcellated/'+scale+'/NAT_MRB_hc77_ding.csv',
                 path+'data/PET_parcellated/'+scale+'/VAChT_feobv_hc3_spreng.csv',
                 path+'data/PET_parcellated/'+scale+'/VAChT_feobv_hc4_tuominen.csv',
                 path+'data/PET_parcellated/'+scale+'/VAChT_feobv_hc5_bedard_sum.csv',
                 path+'data/PET_parcellated/'+scale+'/VAChT_feobv_hc18_aghourian_sum.csv']

# combine all the receptors (including repeats)
r = np.zeros([nnodes, len(receptors_csv)])
for i in range(len(receptors_csv)):
    r[:, i] = np.genfromtxt(receptors_csv[i])[cortex]

receptor_names = np.array(["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                           "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                           "MOR", "NET", "VAChT"])
np.save(path+'data/receptor_names_pet.npy', receptor_names)

# make final region x receptor matrix

receptor_data = np.zeros([nnodes, len(receptor_names)])
receptor_data[:, 0] = r[:, 0]
receptor_data[:, 2:9] = r[:, 3:10]
receptor_data[:, 10:14] = r[:, 12:16]
receptor_data[:, 15:17] = r[:, 19:21]

# weighted average of 5HT1B p943
receptor_data[:, 1] = (r[:, 1]*22 + r[:, 2]*65) / (22+65)

# weighted average of D2 flb457
receptor_data[:, 9] = (r[:, 10]*37 + r[:, 11]*55) / (37+55)

# weighted average of mGluR5 ABP688
receptor_data[:, 14] = (r[:, 16]*22 + r[:, 17]*28 + r[:, 18]*73) / (22+28+73)

# weighted average of VAChT FEOBV
receptor_data[:, 17] = (r[:, 21]*3 + r[:, 22]*4 + r[:, 23] + r[:, 24]) / \
                       (3+4+5+18)

np.savetxt(path+'results/receptor_data_'+scale+'.csv', receptor_data, delimiter=',')


"""
plot receptor data
"""

# colourmaps
cmap = np.genfromtxt(path+'data/colourmap.csv', delimiter=',')
cmap_div = ListedColormap(cmap)

# mean density
annot = datasets.fetch_cammoun2012('fsaverage')[scale]
brain = plotting.plot_fsaverage(data=np.mean(zscore(receptor_data), axis=1),
                                lhannot=annot.lh, rhannot=annot.rh,
                                order = 'rl',
                                colormap=cmap_div,
                                vmin=-np.max(np.abs(np.mean(zscore(receptor_data), axis=1))),
                                vmax=np.max(np.abs(np.mean(zscore(receptor_data), axis=1))),
                                views=['lat', 'med'],
                                data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/surface_recept_density.eps')

# correlating receptors
plt.ion()
plt.figure()
sns.heatmap(np.corrcoef(zscore(receptor_data).T), vmin=-1, vmax=1, cmap=cmap_div,
            cbar=False, square=True, linewidths=.5, xticklabels=receptor_names,
            yticklabels=receptor_names)
plt.tight_layout()
plt.savefig(path+'figures/heatmap_recept_corr.eps')

# correlating regions
plt.ion()
plt.figure()
sns.heatmap(np.corrcoef(zscore(receptor_data)), vmin=-1, vmax=1, cmap=cmap_div,
            cbar=False, square=True, xticklabels=False, yticklabels=False)
plt.tight_layout()
plt.savefig(path+'figures/heatmap_region_corr.eps')

# histogram
plt.ion()
plt.figure()
ax = sns.distplot(np.corrcoef(zscore(receptor_data))[np.triu(np.ones(nnodes), 1) > 0])
ax.set(xlabel = 'receptor similarity')
plt.savefig(path+'figures/hist_receptor_similarity.eps')

#plot each receptor map

annot = datasets.fetch_cammoun2012('fsaverage')['scale125']

for k in range(len(receptor_names)):
    brain = plotting.plot_fsaverage(data=receptor_data[:, k],
                                    lhannot=annot.lh,
                                    rhannot=annot.rh,
                                    order = 'rl',
                                    colormap='plasma',
                                    views=['lat', 'med'],
                                    data_kws={'representation': "wireframe"})
    brain.save_image(path+'figures/surface_receptor_'+receptor_names[k]+'.png')