#-*- coding: utf-8  -*-
"""
Author: michel-pierre.coll
Date: 2020-07-16 09:57:08
Description: Model based multiple regression on ERPs data
TODO:
"""

import mne
from os.path import join as opj
import pandas as pd
import numpy as np
import os
from mne.decoding import Scaler
import scipy
from bids import BIDSLayout
from mne.stats import ttest_1samp_no_p
from mne.stats import spatio_temporal_cluster_1samp_test as st_clust_1s_ttest

###############################
# Parameters
###############################

# Get participants
layout = BIDSLayout('/data/source')
part = ['sub-' + s for s in layout.get_subject()]

# Silence pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

# Parameters
param = {
         # Number of threads for permutations
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
        # TODO remove
        #  # Use FDR (if false witll use TFCE)
        #  'usefdr': True
         }

# Remove excluded part
part = [p for p in part if p not in param['excluded']]

# Outpath for analysis
outpath = '/data/derivatives/statistics/erps_modelbased_ols'
if not os.path.exists(outpath):
    os.makedirs(outpath)

# ########################################################################
# Run multiple regression at the participant level
###########################################################################

# Read computational estimates
mod_data = pd.read_csv('/data/derivatives/task-fearcond_alldata.csv')

# Regressors to include in multiple regression
regvars = ['vhat', 'sa1hat', 'sa2hat']  # Columns in data

# Init empty lists to collect outputs
all_epos = []
allbetasnp = []
betas = [[] for i in range(len(regvars))]

# Loop participants and load single trials file
for p in part:

    # Subset participant
    df = mod_data[mod_data['sub'] == p]

    # Load single epochs file (cotains one epoch/trial)
    epo = mne.read_epochs(opj('/data/derivatives',  p, 'eeg',
                              p + '_task-fearcond_cues_singletrials-epo.fif'))

    # downsample if necessary
    if epo.info['sfreq'] != param['testresampfreq']:
        epo = epo.resample(param['testresampfreq'])

    # Get good trials indices
    goodtrials = np.where(df['badtrial'] == 0)[0]

    # Remove bad trials
    df = df.iloc[goodtrials]
    epo = epo[goodtrials]

    # Standardize data before regression
    # EEG data
    scale = Scaler(scalings='mean')  # Says mean but is z score, see docs
    epo_z = mne.EpochsArray(scale.fit_transform(epo.get_data()),
                            epo.info)

    # computational data
    for regvar in regvars:
        df[regvar + '_z'] = scipy.stats.zscore(df[regvar])

    epo.metadata = df.assign(Intercept=1)  # Add an intercept

    # Perform multiple regression
    no = [r + '_z' for r in regvars]
    names = ["Intercept"] + no
    res = mne.stats.linear_regression(epo_z, epo.metadata[names],
                                        names=names)
    # Get beta values for each regressor
    betasnp = []
    for idx, regvar in enumerate(no):
        betas[idx].append(res[regvar].beta)
        betasnp.append(res[regvar].beta.data)

    allbetasnp.append(np.stack(betasnp))
    all_epos.append(epo)

# Stack all participant data
allbetas = np.stack(allbetasnp)
all_epos = mne.concatenate_epochs(all_epos)

# Grand average in mne class
beta_gavg = []
for idx, regvar in enumerate(no):
    beta_gavg.append(mne.grand_average(betas[idx]))


# ########################################################################
# Perform second level test on betas
###########################################################################

# Find channel adjacency
connect, names = mne.channels.find_ch_adjacency(epo.info,
                                                'eeg')

# Perform test for each regressor
tvals, pvals = [], []
for idx, regvar in enumerate(no):
    # Reshape sub x time x vertices
    testdata = np.swapaxes(allbetas[:, idx, :, :], 2, 1)
    # TFCE
    tval, _, pval, _, = st_clust_1s_ttest(testdata,
                                          n_permutations=param['nperms'],
                                          threshold=dict(start=0,
                                                         step=0.2),
                                          adjacency=connect,
                                          n_jobs=param['njobs'],
                                          seed=param['random_state'])

    # Reshape back to data and append
    tvals.append(np.reshape(tval, (testdata.shape[1],
                            testdata.shape[2])))
    pvals.append(np.reshape(pval, (testdata.shape[1],
                            testdata.shape[2])))

    # Save for each regressor in case crash
    np.save(opj(outpath, 'ols_2ndlevel_tval_' + regvar + '.npy'), tvals[-1])
    np.save(opj(outpath, 'ols_2ndlevel_pval_' + regvar + '.npy'), pvals[-1])

# TODO remove
# # ################################### FDR
# tvals, pvals, sig_clusts = [], [], []
# for idx, regvar in enumerate(no):
#     # Reshape sub x time x vertices
#     testdata = np.swapaxes(allbetas[:, idx, :, :], 2, 1)
#     shape = testdata.shape
#     # Reshape data in a single vector for t-test
#     testdata = testdata.reshape(shape[0], shape[1]*shape[2])
#     # t-test
#     tval = ttest_1samp_no_p(testdata, sigma=1e-3)
#     pval = scipy.stats.t.sf(np.abs(tval), shape[0]-1)*2  # two-sided pvalue

#     # FDR correction
#     _, pval = mne.stats.fdr_correction(pval, alpha=param['alpha'])

#     # Reshape back to data and append
#     tvals.append(np.reshape(tval, (shape[1],
#                                    shape[2])))
#     pvals.append(np.reshape(pval, (shape[1],
#                                    shape[2])))
# ##########################################


# Stack and save
tvals = np.stack(tvals)
pvals = np.stack(pvals)

np.save(opj(outpath, 'ols_2ndlevel_tvals.npy'), tvals)
np.save(opj(outpath, 'ols_2ndlevel_pvals.npy'), pvals)
np.save(opj(outpath, 'ols_2ndlevel_betas.npy'), allbetas)
all_epos.save(opj(outpath, 'ols_2ndlevel_allepochs-epo.fif'), overwrite=True)
np.save(opj(outpath, 'ols_2ndlevel_betasavg.npy'), beta_gavg)
