# Mapping neurotransmitter systems to the structural and functional organization of the human neocortex
This repository contains code and data created in support of my project "Mapping neurotransmitter systems to the structural and functional organization of the human neocortex".
All code was written in Python.
Below, I describe all the folders and files in details.

## `code`
The [code](code/) folder contains all the code used to run the analyses and generate the figures.
If you use this code, don't forget to change the `path` variable.
I regularly use [netneurotools](https://github.com/netneurolab/netneurotools), a handy Python package developed in-house.
This folder contains the following files (in an order that complements the manuscript):
- [parcellate](code/parcellate.py) will parcellate all the PET `nifti` images into the Lausanne parcellation and spit out `csv` files. Code can easily be modified to suit a different parcellation. There's also some surface plotting which uses the handy [neuromaps](https://github.com/netneurolab/neuromaps).
- [make_receptor_matrix.py](code/make_receptor_matrix.py) will load the `csv` files made by [parcellate](code/parcellate.py), take weighted averages of maps that use the same tracer, and save a pre-zscored region-by-receptor matrix. It also covers the plotting in Figure 1 and Figure S1.
- [connectivity.py](code/connectivity.py) covers Figure 2's structure/function analyses. Highlights include a Python-translated version of [Rick's Matlab code](https://www.brainnetworkslab.com/coderesources) for making degree and edge-length preserving null structural connectivity matrices (see [this](https://www.pnas.org/content/115/21/E4880.short) paper).
- [dynamics.py](code/dynamics.py) covers Figure 3's MEG analyses and Figure S2.
- [cognition.py](code/cognition.py) covers Figure 4's receptor-Neurosynth PLS analyses and Figure S5. Highlights include the use of the [pyls](https://github.com/netneurolab/pypyls) Python package to run PLS and code to run distance-dependent cross-validation.
- [disease.py](code/disease.py) covers Figure 5's receptor-ENIGMA analyses and Figure S3. I'm going to put a script up soon that covers how to make [enigma_atrophy.csv](data/enigma_atrophy.csv) but I'm mostly just using the [enigmatoolbox](https://enigma-toolbox.readthedocs.io/en/latest/).
- [autoradiography.py](code/autordiography.py) covers Figure S5 where I replicate Figure 1 and 2 using autoradiography-derived receptor densities. For Figure S6, S7, and S8, I just used the PET-equivalent code with slight modifications to account for the fact that I'm only looking at the left-hemisphere.
- [supplement.py](code/supplement.py) covers Figures S9, S10, and S11, where we compare results using different parcellation resolutions, PET tracers, and test for age effects.

## `data`
The [data](data/) folder contains all the data used to run the analyses.
If you use any of the PET data, please cite (1) the paper(s) that originally collected the data (citations can be found in [Table_S3.xlsx](manuscript/Table_S3.xlsx)) and (2) [this paper (!)]()
Here are the details:
- [PET_nifti_images](data/PET_nifti_images/) has a bunch of group-averaged volumetric PET images. Naming convention is: `<receptor_name>_<tracer>_<number of healthy controls>_<data reference>.nii.gz`. 
- [PET_parcellated](data/PET_parcellated/) has the parcellated PET maps in `csv` format
- [autoradiography](data/autoradiography/) is just a copy of [Al Goulas' repo](https://github.com/AlGoulas/receptor_principles) (in support of [this](https://www.pnas.org/content/118/3/e2020574118.long) paper) but the data originally comes from Karl Zilles and Nicola Palomero-Gallagher's Table S2 in [this](https://www.frontiersin.org/articles/10.3389/fnana.2017.00078/full) paper.
- [lausanne](data/lausanne/) has the structural and functional (group averaged/consensus) networks
- [MEG](data/MEG/) has parcellated group-average power maps for 6 power distributions (order: delta, theta, alpha, beta, low gamma, high gamma) from 33 unrelated subjects in HCP
- [neurosynth](data/neurosynth/) has parcellated Neurosynth association maps for 123 terms from the Cognitive Atlas. See more on how to get these maps from [this course](https://github.com/netneurolab/ipn-summer-school/tree/main/lectures/2021-07-02/13-15) I taught.
- [colourmap.csv](data/colourmap.csv) is my colourmap!
- [enigma_atrophy.csv](data/enigma_atrophy.csv) are the ENIGMA-derived cortical thinning maps (see the [enigmatoolbox](https://enigma-toolbox.readthedocs.io/en/latest/)). Disorder names can be found [here](code/disease.py)
- [mesulam_scale033.csv](data/mesulam_scale033.csv) classify brain regions by the Mesulam classes of laminar differentiaion (see [this](https://github.com/MICA-MNI/micaopen/tree/master/MPC)).
- [receptor_names_pet.npy](data/receptor_names_pet.npy) are the names of the receptors in [receptors_scalexxx.csv](results/).

## `results`
This folder contains the outputs from the analyses _except_ the PLS outputs because they were too large.

## `manuscript`
This folder contains the preprint, supplementary [Table S3](manuscript/Table_S3.xlsx), and eventually the postprint, too.
