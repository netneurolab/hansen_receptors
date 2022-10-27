Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

# Mapping neurotransmitter systems to the structural and functional organization of the human neocortex
This repository contains code and data created in support of my project "Mapping neurotransmitter systems to the structural and functional organization of the human neocortex", now published in [Nature Neuroscience](https://www.nature.com/articles/s41593-022-01186-3) (see [tweeprint](https://twitter.com/misicbata/status/1585653391572832262) here).
All code was written in Python.
Below, I describe all the folders and files in details.

## `code`
The [code](code/) folder contains all the code used to run the analyses and generate the figures.
If you use this code, don't forget to change the `path` variable.
I regularly use [netneurotools](https://github.com/netneurolab/netneurotools), a handy Python package developed in-house.
This folder contains the following files (in an order that complements the manuscript):
- [parcellate](code/parcellate.py) will parcellate all the PET `nifti` images into the Schaefer parcellation and spit out `csv` files. Code can easily be modified to suit a different parcellation.
- [make_receptor_matrix.py](code/make_receptor_matrix.py) will load the `csv` files made by [parcellate](code/parcellate.py), take weighted averages of maps that use the same tracer, and save a pre-zscored region-by-receptor matrix. It also plots all the receptor maps (corresponding to Figure 1 in the paper).
- [rsimilarity.py](code/rsimilarity.py) covers Figure 2's analyses including making the receptor similarity matrix and calculating the first principal component of receptor density.
- [connectivity.py](code/connectivity.py) covers Figure 3's structure/function analyses. Highlights include a Python-translated version of [Rick's Matlab code](https://www.brainnetworkslab.com/coderesources) for making degree and edge-length preserving null structural connectivity matrices (see [this](https://www.pnas.org/content/115/21/E4880.short) paper). This also covers the structure-function coupling analysis and Figure S1, Figure S2.
- [dynamics.py](code/dynamics.py) covers Figure 4's MEG analyses and Figure S3.
- [cognition.py](code/cognition.py) covers Figure 5's receptor-Neurosynth PLS analyses as well as Figure S6. Highlights include the use of the [pyls](https://github.com/netneurolab/pypyls) Python package to run PLS and code to run distance-dependent cross-validation.
- [disease.py](code/disease.py) covers Figure 6's receptor-ENIGMA analyses and Figure S7. The [enigma_atrophy.csv](data/enigma_atrophy.csv) data comes from my [hansen_crossdisorder_vulnerability](https://github.com/netneurolab/hansen_crossdisorder_vulnerability) repo where I mostly use the [enigmatoolbox](https://enigma-toolbox.readthedocs.io/en/latest/).
- [autoradiography.py](code/autordiography.py) covers all the replication analyses using autoradiography-derived receptor density data. This includes Figure S4 (corresponding to Figure 4), Figure S8 (corresponding to Figure 2 and 3), Figure S9 (corresponding to Figure 5), and Figure S10 (corresponding to Figure 6).
- [supplement.py](code/supplement.py) covers Figures S11, S12, and S13, where we compare results using different parcellation resolutions, PET tracers, and test for age effects.

## `data`
The [data](data/) folder contains all the data used to run the analyses.
If you use any of the PET data, please cite (1) the paper(s) that originally collected the data (citations can be found in [Table_S3.xlsx](manuscript/Table_S3.xlsx)) and (2) [this paper (!)](https://www.nature.com/articles/s41593-022-01186-3).
Here are the details:
- [PET_nifti_images](data/PET_nifti_images/) has a bunch of group-averaged volumetric PET images. PET data is also available on [neuromaps](https://github.com/netneurolab/neuromaps), a handy toolbox that makes comparing brain maps and converting brain maps between template spaces and parcellations easy! Naming convention is: `<receptor_name>_<tracer>_<number of healthy controls>_<data reference>.nii.gz`. Note: some of this data was originally available elsewhere. See Copenhagen's [Neurobiology Research Unit](https://xtra.nru.dk/index.html) for serotonin and GABA atlases, and [JuSpace](https://github.com/juryxy/JuSpace) for a collection of PET maps, some of which come from [NeuroVault](https://neurovault.org/). Also: there is one map that is not yet available in volumetric form (`VAChT_feobv_hc3_spreng.nii`) but will be, soon. The parcellated versions are in the [PET_parcellated](data/PET_parcellated/) folder.
- [PET_parcellated](data/PET_parcellated/) has the parcellated PET maps in `csv` format
- [autoradiography](data/autoradiography/) is just a copy of [Al Goulas' repo](https://github.com/AlGoulas/receptor_principles) (in support of [this](https://www.pnas.org/content/118/3/e2020574118.long) paper) but the data originally comes from Karl Zilles and Nicola Palomero-Gallagher's Table S2 in [this](https://www.frontiersin.org/articles/10.3389/fnana.2017.00078/full) paper.
- [lausanne](data/lausanne/) has the structural and functional (group averaged/consensus) networks for the Lausanne atlas, which is the atlas I used for the preprint (but ended up replacing this with Schaefer during the review process).
- [schaefer](data/schaefer/) has the structural and functional (group averaged/consensus) networks for the Schaefer atlas.
- [MEG](data/MEG/) has parcellated group-average power maps for 6 power distributions (order: delta, theta, alpha, beta, low gamma, high gamma) from 33 unrelated subjects in HCP. Note that HCP data redistribution must follow their [data terms](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms), including registration with ConnectomeDB, agreement of their terms, and registration for Open Access Data [here](https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms). Please also cite relevant publications as mentioned [here](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms).
- [neurosynth](data/neurosynth/) has parcellated Neurosynth association maps for 123 terms from the Cognitive Atlas. See more on how to get these maps from [this course](https://github.com/netneurolab/ipn-summer-school/tree/main/lectures/2021-07-02/13-15) I taught.
- [colourmap.csv](data/colourmap.csv) is my colourmap!
- [enigma_atrophy.csv](data/enigma_atrophy.csv) are the ENIGMA-derived cortical thinning maps (see the [enigmatoolbox](https://enigma-toolbox.readthedocs.io/en/latest/)). Disorder names can be found [here](code/disease.py)
- [mesulam_scale033.csv](data/mesulam_scale033.csv) classify brain regions by the Mesulam classes of laminar differentiaion (see [this](https://github.com/MICA-MNI/micaopen/tree/master/MPC)).
- [receptor_names_pet.npy](data/receptor_names_pet.npy) are the names of the receptors in [receptors_scalexxx.csv](results/).

## `results`
This folder contains the outputs from the analyses _except_ the PLS outputs because they were too large.

## `manuscript`
This folder contains paper, the preprint, and supplementary [Table S3](manuscript/Table_S3.xlsx).
