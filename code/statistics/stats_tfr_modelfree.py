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
# from mne.stats import spatio_temporal_cluster_1samp_test as perm1samp
# from functools import partial


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

        pdat.append(np.float32(data[cond][-1].data))

    anova_data.append(np.stack(pdat))
    gavg[cond] = mne.grand_average(data[cond])


anova_data = np.stack(anova_data)

# # Take difference of interest for each part
diff_data = np.empty((1,) + anova_data.shape[1:])

for s in range(anova_data.shape[1]):
    diff_data[0, s, ::] = ((anova_data[0, s, :] - anova_data[1, s, :])
                           - (anova_data[2, s, :] - anova_data[3, s, :]))

diff_data.shape
diff_data = np.squeeze(diff_data)

# #########################################################################
# ANOVA
##########################################################################
# Always output time x freq x chan


#  TFCE
#
# chan_connect, _ = mne.channels.find_ch_connectivity(data['CS-1'][0].info,
#                                                     'eeg')
# # Create a 1 ajacent frequency connectivity
# freq_connect = (np.eye(len(data['CS-1'][0].freqs))
#                 + np.eye(len(data['CS-1'][0].freqs), k=1)
#                 + np.eye(len(data['CS-1'][0].freqs), k=-1))
#
# # Combine matrices to get a freq x chan connectivity matrix
# connect = scipy.sparse.csr_matrix(np.kron(freq_connect,
#                                           chan_connect.toarray())
#                                   + np.kron(freq_connect,
#                                             chan_connect.toarray()))
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

tval = ttest_1samp_no_p(diff_data, sigma=1e-3)
pval = scipy.stats.t.sf(np.abs(tval), shapet[0]-1)*2  # two-sided pvalue
_, pval = mne.stats.fdr_correction(pval, alpha=param['alpha'])

# ##############################################################
# TFCE
# from mne.stats import spatio_temporal_cluster_1samp_test as perm1samp
# from functools import partial
#
# # T-test with hat correction
# stat_fun_hat = partial(ttest_1samp_no_p, sigma=1e-3)
#
# testdata = diff_data.swapaxes(1, 3)
# shapet = testdata.shape
# testdata = np.reshape(diff_data, (shapet[0],
#                                   shapet[1], shapet[2]*shapet[3]))
#
# # data is (n_observations, n_times, n_vertices)
# tval, _, pval, _ = perm1samp(testdata,
#                              n_jobs=param['njobs'],
#                              threshold=dict(start=0, step=0.2),
#                              connectivity=connect,
#                              max_step=1,
#                              n_permutations=param['nperms'],
#                              buffer_size=None)

# ##############################################################


# Reshape in chan x freq x time
pvals = np.reshape(pval, (shapet[1],
                          shapet[2],
                          shapet[3]))
tvals = np.reshape(tval, (shapet[1],
                          shapet[2],
                          shapet[3]))


np.save(opj(outpath, 'cuesdiff_tfr_ttest_pvals.npy'), pvals)
np.save(opj(outpath, 'cuesduff_tfr_ttest_tvals.npy'), tvals)
np.save(opj(outpath, 'resamp_times.npy'), data['CS+'][0].times)
np.save(opj(outpath, 'resamp_freqs.npy'), data['CS+'][0].freqs)
