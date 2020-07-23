# #########################################################################
# ERP analyses for Zoey's conditioning task
# @MP Coll, 2020, michelpcoll@gmail.com
##############################################################################

import mne
import pandas as pd
import numpy as np
import os
from os.path import join as opj
from bids import BIDSLayout
from mne.stats import ttest_1samp_no_p
import scipy.stats

###############################
# Parameters
###############################
layout = BIDSLayout('/data/source')
part = ['sub-' + s for s in layout.get_subject()]

# Remove stupid pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

# Outpath for analysis
outpath = '/data/derivatives/statistics/erps_modelfree_anova'
if not os.path.exists(outpath):
    os.makedirs(outpath)

param = {


         # Njobs for permutations
         'njobs': 20,
         # Alpha Threshold
         'alpha': 0.05,
         # Number of permutations
         'nperms': 5000,
         # Random state to get same permutations each time
         'random_state': 23,
         # Downsample to this frequency prior to analysis
         'testresampfreq': 256,
         # excluded participants
         'excluded': ['sub-24', 'sub-31', 'sub-35', 'sub-51'],
         # Use FDR (if false witll use TFCE)
         'usefdr': True
         }

part = [p for p in part if p not in param['excluded']]

###########################################################################
# Load and stack data
###########################################################################

# Epoched data
conditions = ['CS-1', 'CS-2', 'CS+', 'CS-E']

data = dict()
gavg = dict()
for cond in conditions:
    data[cond] = list()
    for p in part:
        outdir = opj('/data/derivatives',  p, 'eeg')

        data[cond].append(mne.read_evokeds(opj(outdir,
                                               p + '_task-fearcond_' + cond
                                               + '_ave.fif'))[0])

#####################################################################
# Statistics - T-test on the difference between CS+ vs CS-E and CS-1 vs CS-2
#####################################################################

# Stack data for ANOVA to get a (cond x subs x  x time x chans) array

anova_data = list()
for idxc, cond in enumerate(conditions):
    cond_data = []
    for idxp, p in enumerate(part):
        pdat = data[cond][idxp].copy()
        # RESAMPLE FOR ANALYSES
        if pdat.info['sfreq'] != param['testresampfreq']:
            pdat = pdat.resample(param['testresampfreq'], npad='auto')

        cond_data.append(np.swapaxes(pdat.data, axis1=1, axis2=0))
    anova_data.append(np.stack(cond_data))

anova_data = np.stack(anova_data)


# # Take difference of interest for each part
csplusvscs1 = np.empty((1,) + anova_data.shape[1:])
csevscs2 = np.empty((1,) + anova_data.shape[1:])
csplusvscse = np.empty((1,) + anova_data.shape[1:])
csplusvscse2 = np.empty((1,) + anova_data.shape[1:])

# Calculate differences
for s in range(anova_data.shape[1]):

    csplusvscs1[0, s, ::] = (anova_data[2, s, :] - anova_data[0, s, :])
    csevscs2[0, s, ::] = (anova_data[3, s, :] - anova_data[1, s, :])
    csplusvscse[0, s, ::] = ((anova_data[2, s, :] - anova_data[3, s, :]) - (anova_data[0, s, :] - anova_data[1, s, :]))

csplusvscs1 = np.squeeze(csplusvscs1)
csevscs2 = np.squeeze(csevscs2)
csplusvscse = np.squeeze(csplusvscse)

shape = csplusvscs1.shape

# # # TFCE
from mne.stats import spatio_temporal_cluster_1samp_test as perm1samp
from functools import partial
# Get channels connectivity
connect, names = mne.channels.find_ch_adjacency(data['CS+'][0].info,
                                                'eeg')
# T-test with hat correction

# data is (n_observations, n_times, n_vertices)
tval, _, pval, _ = perm1samp(csplusvscse,
                             n_jobs=param["njobs"],
                             threshold=dict(start=0, step=0.2),
                             connectivity=connect,
                             n_permutations=param['nperms'],
                             buffer_size=None)

pvals = np.reshape(pval, (shape[1],
                          shape[2]))
tvals = np.reshape(tval, (shape[1],
                          shape[2]))

np.save(opj(outpath, 'csplusvscse_ttest_pvals.npy'), pvals)
np.save(opj(outpath, 'csplusvscse_ttest_tvals.npy'), tvals)
np.save(opj(outpath, 'resamp_times.npy'), pdat.times)


tval, _, pval, _ = perm1samp(csplusvscs1,
                             n_jobs=param["njobs"],
                             threshold=dict(start=0, step=0.2),
                             connectivity=connect,
                             n_permutations=param['nperms'],
                             buffer_size=None)

pvals = np.reshape(pval, (shape[1],
                          shape[2]))
tvals = np.reshape(tval, (shape[1],
                          shape[2]))

np.save(opj(outpath, 'csplusvscs1_ttest_pvals.npy'), pvals)
np.save(opj(outpath, 'csplusvscs1_ttest_tvals.npy'), tvals)


tval, _, pval, _ = perm1samp(csevscs2,
                             n_jobs=param["njobs"],
                             threshold=dict(start=0, step=0.2),
                             connectivity=connect,
                             n_permutations=param['nperms'],
                             buffer_size=None)

pvals = np.reshape(pval, (shape[1],
                          shape[2]))
tvals = np.reshape(tval, (shape[1],
                          shape[2]))

np.save(opj(outpath, 'csevscs2_ttest_pvals.npy'), pvals)
np.save(opj(outpath, 'csevscs2_ttest_tvals.npy'), tvals)
