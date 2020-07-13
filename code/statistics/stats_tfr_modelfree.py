# #########################################################################
# TFR analyses for Zoey's conditioning task
# @MP Coll, 2020, michelpcoll@gmail.com
##############################################################################

import mne
import pandas as pd
import numpy as np
import os
from os.path import join as opj
from bids import BIDSLayout
from mne.stats import ttest_1samp_no_p
from mne.time_frequency import read_tfrs
import scipy
from mne.stats import spatio_temporal_cluster_1samp_test as perm1samp

# from mne.stats import spatio_temporal_cluster_1samp_test as perm1samp
# !pip install git+https://github.com/larsoner/mne-python@conn
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

outpath = '/data/derivatives/statistics/tfr_modelfree_anova'
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
         # excluded participants
         'excluded': ['sub-24', 'sub-31', 'sub-35', 'sub-51'],

         }

part = [p for p in part if p not in param['excluded']]


###########################################################################
# Load and stack data
###########################################################################

conditions = ['CS-1', 'CS-2', 'CS+', 'CS-E', ]
anova_data = []
data = dict()
gavg = dict()
for cond in conditions:
    pdat = []
    data[cond] = []

    for p in part:
        data[cond].append(read_tfrs(opj('/data/derivatives', p,
                                        'eeg',
                                        p + '_task-fearcond_' + cond
                                        + '_avg-tfr.h5'))[0])

        data[cond][-1].apply_baseline(mode='logratio',
                                      baseline=(-0.2, 0))

        data[cond][-1].crop(tmin=0, tmax=1)

        pdat.append(np.float32(data[cond][-1].data))

    anova_data.append(np.stack(pdat))
    gavg[cond] = mne.grand_average(data[cond])


anova_data = np.stack(anova_data)

# # Take difference of interest for each part
diff_data = np.empty((1,) + anova_data.shape[1:])
diff1 = np.empty((1,) + anova_data.shape[1:])
diff2 = np.empty((1,) + anova_data.shape[1:])

for s in range(anova_data.shape[1]):
    diff_data[0, s, :, :, :] = ((anova_data[0, s, :, :, :] - anova_data[1, s, :, :, :])
                                - (anova_data[2, s, :, :, :] - anova_data[3, s, :, :, :]))


diff_data = np.squeeze(diff_data)
diff1 = np.squeeze(diff1)
diff2 = np.squeeze(diff2)
#
# # #########################################################################
# # ANOVA
# ##########################################################################
# # Always output time x freq x chan
#
# #  TFCE

#
#
# from mne.stats import permutation_cluster_1samp_test
#
#
# def stat_fun(*args):  # Custom ANOVA for permutation
#     return mne.stats.f_mway_rm(np.swapaxes(args, 1, 0),  # Swap sub and cond
#                                factor_levels=[4],
#                                effects='A',
#                                return_pvals=False,
#                                correction=True)[0]
#
#
# tfce_dict = dict(start=0, step=0.2)
#
# # Reshape in cond, sub, time*freq, chan
# anova_data_test = anova_data.swapaxes(2, 4)
# shapea = anova_data_test.shape
# anova_data_test = np.reshape(anova_data_test,
#                              (shapea[0], shapea[1], shapea[2],
#                               shapea[3]*shapea[4]))
#
# F_obs, clusters, pval, h0 = \
#     mne.stats.permutation_cluster_test(anova_data_test,
#                                        stat_fun=stat_fun,
#                                        threshold=tfce_dict,
#                                        tail=1,  # One tail cause anova
#                                        n_permutations=param['nperms'],
#                                        seed=23,
#                                        connectivity=connect,
#                                        max_step=1,
#                                        n_jobs=param['njobs'],
#                                        out_type='indices')

# ##########################################################################
# FDR
shapea = anova_data.shape
anova_data_uni = np.reshape(anova_data, (shapea[0], shapea[1],
                                         shapea[2]*shapea[3]*shapea[4]))
F_obs, pval = mne.stats.f_mway_rm(np.swapaxes(anova_data_uni, 1, 0),
                                  factor_levels=[4],
                                  effects='A',
                                  return_pvals=True,
                                  correction=True)
_, pval = mne.stats.fdr_correction(pval, alpha=param['alpha'])

pvals = np.reshape(pval, (shapea[2],
                          shapea[3],
                          shapea[4]))
F_obsout = np.reshape(F_obs, (shapea[2],
                              shapea[3],
                              shapea[4]))

np.save(opj(outpath, 'cues4_tfr_anova_pvals.npy'), pvals)
np.save(opj(outpath, 'cues4_tfr_anova_Fvals.npy'), F_obsout)
np.save(opj(outpath, 'resamp_times.npy'), data['CS+'][0].times)
np.save(opj(outpath, 'resamp_freqs.npy'), data['CS+'][0].freqs)

# Difference
# ##############################################################
# FDR
shapet = diff_data.shape
testdata = np.reshape(diff_data, (shapet[0], shapet[1]*shapet[2]*shapet[3]))

#
tval = ttest_1samp_no_p(testdata, sigma=1e-3)
pval = scipy.stats.t.sf(np.abs(tval), shapet[0]-1)*2  # two-sided pvalue
#
_, pval = mne.stats.fdr_correction(pval, alpha=param['alpha'])


# ##############################################################
# TFCE
# from functools import partial
#
# T-test with hat correction
testdata = np.swapaxes(diff_data, 1, 3)
shapet = testdata.shape


chan_connect, _ = mne.channels.find_ch_adjacency(data['CS-1'][0].info, 'eeg')


connectivity = mne.stats.combine_adjacency(len(data['CS-1'][0].freqs),
                                           chan_connect)

# !pip install git+https://github.com/larsoner/mne-python@conn
# data is (n_observations, n_times*n_vertices)
testdata = np.reshape(testdata, (shapet[0], shapet[1], shapet[2]*shapet[3]))
tval, clusters, pval, H0 = perm1samp(testdata,
                                     n_jobs=param['njobs'],
                                     threshold=dict(start=0, step=0.5),
                                     connectivity=connectivity,
                                     max_step=1,
                                     check_disjoint=True,
                                     n_permutations=2000,
                                     buffer_size=None)

tvals = np.reshape(tval, (shapet[1],
                          shapet[2],
                          shapet[3]))
tvals = np.swapaxes(tvals, 0, 2)

pvals = np.reshape(pval, (shapet[1],
                          shapet[2],
                          shapet[3]))
pvals = np.swapaxes(pvals, 0, 2)

dat = data[cond][-1].copy()
dat.data = tvals
dat.plot_topomap()
print(np.max(np.abs(tvals)))

dat = data[cond][-1].copy()
dat.data = np.where(pvals < 0.05, 1, 0)
dat.plot_topomap()
dat.plot('POz')
print(np.min(np.abs(pvals)))

# ##############################################################
# Reshape in chan x freq x time

np.save(opj(outpath, 'cuesdiff_tfr_ttest_pvals.npy'), pvals)
np.save(opj(outpath, 'cuesdiff_tfr_ttest_tvals.npy'), tvals)
np.save(opj(outpath, 'resamp_times.npy'), data['CS+'][0].times)
np.save(opj(outpath, 'resamp_freqs.npy'), data['CS+'][0].freqs)
