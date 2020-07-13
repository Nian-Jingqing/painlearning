# #########################################################################
# Model based regression on ERPs data
# @MP Coll, 2020, michelpcoll@gmail.com
# #########################################################################

import mne
from os.path import join as opj
import pandas as pd
import numpy as np
import os
from mne.decoding import Scaler
import scipy
from bids import BIDSLayout
from mne.stats import ttest_1samp_no_p

###############################
# Parameters
###############################
layout = BIDSLayout('/data/source')
part = ['sub-' + s for s in layout.get_subject()]

# Silence pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

# Parameters
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

# Outpath for analysis
outpath = '/data/derivatives/statistics'
if not os.path.exists(outpath):
    os.mkdir(outpath)

outpath = '/data/derivatives/statistics/erps_modelbased_ols_single'
if not os.path.exists(outpath):
    os.mkdir(outpath)


# ########################################################################
# Run linear models at the first level
###########################################################################

mod_data = pd.read_csv('/data/derivatives/task-fearcond_alldata.csv')

regvars = ['vhat', 'sa1hat', 'sa2hat']
regvarsnames = ['Expectation', 'Irr. uncertainty', 'Est. uncertainty']

betas, betasnp = [], []

# Loop participants and load single trials file
all_epos = []
allbetasnp = []
betas = [[] for i in range(len(regvars))]
for p in part:
    df = mod_data[mod_data['sub'] == p]

    # Load single epochs file (cotains one epoch/trial)
    epo = mne.read_epochs(opj('/data/derivatives',  p, 'eeg',
                              p + '_task-fearcond_cues_singletrials-epo.fif'))

    # downsample if necessary
    if epo.info['sfreq'] != param['testresampfreq']:
        epo = epo.resample(param['testresampfreq'])

    # Drop bad trials and get indices
    goodtrials = np.where(df['badtrial'] == 0)

    # Get external data for this part
    df = df.iloc[goodtrials]

    epo = epo[goodtrials[0]]
    # Bin trials by value and plot GFP

    # Standardize data before regression
    # EEG data
    scale = Scaler(scalings='mean')  # Says mean but is z score, see docs
    epo_z = mne.EpochsArray(scale.fit_transform(epo.get_data()),
                            epo.info)

    # Standardize data
    betasnp = []
    for idx, regvar in enumerate(regvars):
        df[regvar + '_z'] = scipy.stats.zscore(df[regvar])

        epo.metadata = df.assign(Intercept=1)  # Add an intercept for later

        names = ["Intercept"] + [regvar + '_z']
        res = mne.stats.linear_regression(epo_z, epo.metadata[names],
                                          names=names)

        betas[idx].append(res[regvar + '_z'].beta)
        betasnp.append(res[regvar + '_z'].beta.data)

    allbetasnp.append(np.stack(betasnp))
    all_epos.append(epo)

# Stack all data
allbetas = np.stack(allbetasnp)
all_epos = mne.concatenate_epochs(all_epos)

# Grand average
beta_gavg = []
for idx, regvar in enumerate(regvars):
    beta_gavg.append(mne.grand_average(betas[idx]))


# ########################################################################
# Perform second level test on betas
###########################################################################

# # TFCE
# from mne.stats import spatio_temporal_cluster_1samp_test as st_clust_1s_ttest

# connect, names = mne.channels.find_ch_connectivity(epo.info,
#                                                    'eeg')
# tvals, pvals, sig_clusts = [], [], []
# for idx, regvar in enumerate(no):
#     # Reshape sub x time x vertices
#     testdata = np.swapaxes(allbetas[:, idx, :, :], 2, 1)
#     tval, _, pval, _, = st_clust_1s_ttest(testdata,
#                                           n_permutations=param['nperms'],
#                                           threshold=dict(start=0,
#                                                          step=0.2),
#                                           connectivity=connect,
#                                           n_jobs=param['njobs'],
#                                           seed=param['random_state'])
#
#     # Reshape back to data and append
#     tvals.append(np.reshape(tval, (testdata.shape[1],
#                             testdata.shape[2])))
#     pvals.append(np.reshape(pval, (testdata.shape[1],
#                             testdata.shape[2])))


# ################################### FDR
tvals, pvals, sig_clusts = [], [], []
for idx, regvar in enumerate(regvars):
    # Reshape sub x time x vertices
    testdata = np.swapaxes(allbetas[:, idx, :, :], 2, 1)
    shape = testdata.shape
    # Reshape data in a single vector for t-test
    testdata = testdata.reshape(shape[0], shape[1]*shape[2])
    # t-test
    tval = ttest_1samp_no_p(testdata, sigma=1e-3)
    pval = scipy.stats.t.sf(np.abs(tval), shape[0]-1)*2  # two-sided pvalue

    # FDR correction
    _, pval = mne.stats.fdr_correction(pval, alpha=param['alpha'])

    # Reshape back to data and append
    tvals.append(np.reshape(tval, (shape[1],
                                   shape[2])))
    pvals.append(np.reshape(pval, (shape[1],
                                   shape[2])))
##########################################


# Stack and save
tvals = np.stack(tvals)
pvals = np.stack(pvals)

np.save(opj(outpath, 'ols_2ndlevel_tvals.npy'), tvals)
np.save(opj(outpath, 'ols_2ndlevel_pvals.npy'), pvals)
np.save(opj(outpath, 'ols_2ndlevel_betas.npy'), allbetas)
all_epos.save(opj(outpath, 'ols_2ndlevel_allepochs-epo.fif'))
np.save(opj(outpath, 'ols_2ndlevel_betasavg.npy'), beta_gavg)
