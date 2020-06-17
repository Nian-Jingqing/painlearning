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
# from mne.stats import spatio_temporal_cluster_1samp_test as perm1samp
# from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from mne.decoding import Vectorizer
from mne.time_frequency import read_tfrs
import scipy
from functools import partial

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

outpath = '/data/derivatives/statistics/tfr_modelbased_ols'
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
         'testresampfreq': 128,
         # excluded participants
         'excluded': ['sub-24', 'sub-31', 'sub-35', 'sub-51'],

         }

part = [p for p in part if p not in param['excluded']]


# ########################################################################
# Run linear models at the first level
###########################################################################
mod_data = pd.read_csv('/data/derivatives/task-fearcond_alldata.csv')

regvars = ['vhat', 'sa1hat', 'sa2hat']
regvarsnames = ['Expectation', 'Irr. uncertainty', 'Est. uncertainty']

# betas, betasnp = [], []
#
# # Loop participants and load single trials file
# allbetasnp, all_epos = [], []
#
# for p in part:
#
#     df = mod_data[mod_data['sub'] == p]
#
#     # Load single epochs file (cotains one epoch/trial)
#     epo = read_tfrs(opj('/data/derivatives',  p, 'eeg',
#                         p + '_task-fearcond_epochs-tfr.h5'))[0]
#
#     # Drop bad trials and get indices
#     goodtrials = np.where(df['badtrial'] == 0)
#
#     # Get external data for this part
#
#     if len(df) != 0:
#         df = df.iloc[goodtrials]
#
#         epo = epo[goodtrials[0]]
#         # Bin trials by value and plot GFP
#
#         # Standardize data before regression
#         # EEG data
#         # Vectorize, Zscore,
#         clf = make_pipeline(Vectorizer(),
#                             StandardScaler(),
#                             LinearRegression(n_jobs=param['njobs']))
#
#         clf.fit(epo.data, df['sa1hat'])
#
#         # Standardize data
#         for regvar in regvars:
#             df[regvar + '_z'] = ((df[regvar]-np.average(df[regvar]))
#                                  / np.std(df[regvar]))
#
#         no = [r + '_z' for r in regvars]
#         # Fit regression
#         clf.fit(epo.data, df[no])
#
#         betasnp = []
#         for idx, regvar in enumerate(no):
#             out = np.reshape(clf['linearregression'].coef_[idx],
#                              (epo.data.shape[1], epo.data.shape[2],
#                               epo.data.shape[3]))
#             betasnp.append(out)
#
#         allbetasnp.append(np.stack(betasnp))
#
#
# # Stack all data
# allbetas = np.stack(allbetasnp)
#
# # Create MNE EpochsArray
#
# np.save(opj(outpath, 'ols_2ndlevel_allbetas.npy'), allbetas)

# Grand average
# #########################################################################
# Perform second level test on betas
##########################################################################
# Always output time x freq x chan

allbetas = np.load(opj(outpath, 'ols_2ndlevel_allbetas.npy'))
epo = read_tfrs(opj('/data/derivatives',  part[0], 'eeg',
                    part[0] + '_task-fearcond_epochs-tfr.h5'))[0]

stat_fun_hat = partial(ttest_1samp_no_p, sigma=1e-3)
chan_connect, _ = mne.channels.find_ch_connectivity(epo.info,
                                                    'eeg')
# Create a 1 ajacent frequency connectivity
freq_connect = (np.eye(len(epo.freqs)) + np.eye(len(epo.freqs), k=1)
                + np.eye(len(epo.freqs), k=-1))

# Combine matrices to get a freq x chan connectivity matrix
connect = scipy.sparse.csr_matrix(np.kron(freq_connect,
                                          chan_connect.toarray())
                                  + np.kron(freq_connect,
                                            chan_connect.toarray()))

tvals, pvals, sig_clusts = [], [], []
for idx, regvar in enumerate(regvars):

    # # # TFCE
    # betas_tfce = allbetas[:, idx, ::]
    # from mne.stats import spatio_temporal_cluster_1samp_test as clust_1s_ttest
    # betas_tfce = np.reshape(betas_tfce, (betas_tfce.shape[0],
    #                                      betas_tfce.shape[1]
    #                                      * betas_tfce.shape[2],
    #                                      betas_tfce.shape[3]))
    # # Make channels first axis
    # # TFCE dict
    # tfce = dict(start=0, step=0.2)
    # betas_tfce = np.swapaxes(betas_tfce, 1, 2)
    # betas_tfce.shape
    # tval, _, pval, _, = clust_1s_ttest(betas_tfce,
    #                                    n_permutations=param['nperms'],
    #                                    threshold=dict(start=0,
    #                                                   step=0.2),
    #                                    connectivity=connect,
    #                                    stat_fun=stat_fun_hat,
    #                                    max_step=1,
    #                                    buffer_size=None,
    #                                    n_jobs=param['njobs'],
    #                                    seed=param['random_state'])

    # Reshape back to data and append
    # tvals.append(np.reshape(tval, (betas_tfce.shape[1],
    #                                betas_tfce.shape[2],
    #                                betas_tfce.shape[3])
    #                         ))
    # pvals.append(np.reshape(pval, (betas_tfce.shape[1],
    #                                betas_tfce.shape[2],
    #                                betas_tfce.shape[3])
    #                         ))

    # FDR
    # Reshape data in a single vector for t-test

    # t-test
    testdata = allbetas[:, idx, ::]
    shapet = testdata.shape
    testdata = testdata.reshape(shapet[0], shapet[1]*shapet[2]*shapet[3])
    tval = ttest_1samp_no_p(testdata, sigma=1e-3)
    pval = scipy.stats.t.sf(np.abs(tval), shapet[0]-1)*2  # two-sided pvalue

    # FDR correction
    _, pval = mne.stats.fdr_correction(pval, alpha=param['alpha'])

    tvals.append(np.reshape(tval, shapet[1:]))
    pvals.append(np.reshape(pval, shapet[1:]))

tvals = np.stack(tvals)
pvals = np.stack(pvals)

np.save(opj(outpath, 'ols_2ndlevel_tfr_pvals.npy'), pvals)
np.save(opj(outpath, 'ols_2ndlevel_tfr_tvals.npy'), tvals)
np.save(opj(outpath, 'resamp_times.npy'), epo.times)
np.save(opj(outpath, 'resamp_freqs.npy'), epo.freqs)
