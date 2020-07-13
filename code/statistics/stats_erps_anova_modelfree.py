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
outpath = '/data/derivatives/statistics'
if not os.path.exists(outpath):
    os.mkdir(outpath)

outpath = '/data/derivatives/statistics/erps_modelfree_anova'
if not os.path.exists(outpath):
    os.mkdir(outpath)

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
         'testresampfreq': 1024,
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
# Statistics - T-test on the difference between CS+ vs CS-1 and CS-E vs CS-2
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
diff_data = np.empty((1,) + anova_data.shape[1:])
diff_data.shape
for s in range(anova_data.shape[1]):
    diff_data[0, s, ::] = ((anova_data[0, s, :] - anova_data[1, s, :])
                           - (anova_data[2, s, :] - anova_data[3, s, :]))

diff_data = np.squeeze(diff_data)
shape = diff_data.shape


# # # # TFCE
# from mne.stats import spatio_temporal_cluster_1samp_test as perm1samp
# from functools import partial
# # Get channels connectivity
# connect, names = mne.channels.find_ch_connectivity(data['CS+'][0].info,
#                                                    'eeg')
# # T-test with hat correction
# stat_fun_hat = partial(ttest_1samp_no_p, sigma=1e-3)
#
# # data is (n_observations, n_times, n_vertices)
# tval, _, pval, _ = perm1samp(diff_data,
#                              n_jobs=param["njobs"],
#                              threshold=dict(start=0, step=0.2),
#                              connectivity=connect,
#                              stat_fun=stat_fun_hat,
#                              n_permutations=param['nperms'],
#                              buffer_size=None)

# #############################################################################
# FDR
# Reshape data in a single vector
testdata = np.reshape(diff_data, (shape[0], shape[1]*shape[2]))
tval = ttest_1samp_no_p(testdata, sigma=1e-3)
pval = scipy.stats.t.sf(np.abs(tval), shape[0]-1)*2  # two-sided pvalue
_, pval = mne.stats.fdr_correction(pval, alpha=param['alpha'])
# ###########################################################################

# #############################################################################
# Permutation
# Reshape data in a single vector
# shape = diff_data.shape
# testdata = np.reshape(diff_data, (shape[0], shape[1]*shape[2]))
# tval, pval, _ = permutation_t_test(testdata, n_permutations=param['nperms'],
#                                    n_jobs=param['njobs'])

# ###########################################################################


# Reshape in time x chan
pvals = np.reshape(pval, (shape[1],
                          shape[2]))
tvals = np.reshape(tval, (shape[1],
                          shape[2]))

np.save(opj(outpath, 'cuesdiff_ttest_pvals.npy'), pvals)
np.save(opj(outpath, 'cuesdiff_ttest_tvals.npy'), tvals)
np.save(opj(outpath, 'resamp_times.npy'), pdat.times)

#
# # TFCE
# # Same thing but using a 4-way ANOVA instead
# def stat_fun(*args):  # Custom ANOVA for permutation
#     return mne.stats.f_mway_rm(np.swapaxes(args, 1, 0),  # Swap sub and cond
#                                factor_levels=[4],
#                                effects='A',
#                                return_pvals=False,
#                                correction=True)[0]
#
#
# # Use TFCE
# tfce_dict = dict(start=0, step=0.2)
# # Run the permuted ANOVA
# shape = anova_data.shape
# F_obs, clusters, pvals, h0 = \
#     mne.stats.spatio_temporal_cluster_test(anova_data,
#                                            stat_fun=stat_fun,
#                                            threshold=tfce_dict,
#                                            connectivity=connect,
#                                            tail=1,  # One tail cause anova
#                                            n_permutations=param['nperms'],
#                                            seed=param['random_state'],
#                                            max_step=1,
#                                            check_disjoint=True,
#                                            n_jobs=param['njobs'],
#                                            out_type='mask')

# ######################## FDR #######################################
anova_data_uni = np.swapaxes(np.stack(anova_data), 1, 0)
shape = anova_data_uni.shape
anova_data_uni = np.reshape(anova_data_uni, (shape[0], shape[1],
                                             shape[2]*shape[3]))

F_obs, pvals = mne.stats.f_mway_rm(anova_data_uni,  # Swap sub and cond
                                   factor_levels=[4],
                                   effects='A',
                                   return_pvals=True,
                                   correction=True)
_, pvals = mne.stats.fdr_correction(pvals, alpha=param['alpha'])
# ######################## FDR #######################################


# Reshape time x freq
pvals = np.reshape(pvals, (shape[2],
                           shape[3]))
F_obsout = np.reshape(F_obs, (shape[2],
                              shape[3]))

np.save(opj(outpath, 'cues4_anova_pvals.npy'), pvals)
np.save(opj(outpath, 'cues4_anova_Fvals.npy'), F_obsout)
np.save(opj(outpath, 'resamp_times.npy'), pdat.times)
