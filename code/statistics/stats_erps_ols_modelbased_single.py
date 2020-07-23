#-*- coding: utf-8  -*-
"""
Author: michel-pierre.coll
Date: 2020-07
Description: Perform multivariate regression on ERP data
"""

import mne
from os.path import join as opj
import pandas as pd
import numpy as np
import os
from mne.decoding import Scaler
import scipy
from bids import BIDSLayout
from mne.stats import spatio_temporal_cluster_1samp_test as st_clust_1s_ttest
import scipy.stats
import statsmodels.api as sm
from oct2py import octave
from scipy.stats import zscore
import pickle

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
         # Number of permutations
         'nperms': 5000,
         # Random state to get same permutations each time
         'random_state': 23,
         # Downsample to this frequency prior to analysis
         'testresampfreq': 256,
         # excluded participants
         'excluded': ['sub-24', 'sub-31', 'sub-35', 'sub-51'],
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
# Load erps and data
###########################################################################
# Read computational estimates
mod_data = pd.read_csv('/data/derivatives/task-fearcond_alldata.csv')

# Regressors to include in multiple regression
regvars = ['vhat', 'sa1hat', 'sa2hat']  # Columns in data

# Init empty lists to collect outputs
all_epos = []
allbetasnp = []
betas = [[] for i in range(len(regvars))]

bic_erps = pd.DataFrame(index=part, data={'vhat': 999,
                                          'sa1hat': 999,
                                          'sa2hat': 999})

bic_beta = pd.DataFrame(index=part, data={'vhat': 999,
                                          'sa1hat': 999,
                                          'sa2hat': 999})
# Loop participants and load single trials file


for variable in ['vhat', 'sa1hat', 'sa2hat']:
    for p in part:

        # Subset participant
        df = mod_data[mod_data['sub'] == p]

        # Get good trials indices
        goodtrials = np.where(df['badtrial'] == 0)[0]

        df = df.iloc[goodtrials]


        mod = sm.OLS(df["amp_['POz']_0.4-0.8"],  sm.add_constant(df[variable]),
                    hasconst=True).fit()

        bic_erps.loc[p, variable] = mod.bic
        bic_beta.loc[p, variable] = mod.params[variable]

bic_erps.to_csv(opj(outpath, 'bic_erps.csv'))

# Use octave to run the VBA-toolbox
octave.push('L', np.asarray(bic_erps.transpose())*-1)
octave.addpath('/matlab/vbatoolbox')
octave.addpath('/matlab/vbatoolbox/core')
octave.addpath('/matlab/vbatoolbox/core/display')
octave.addpath('/matlab/vbatoolbox/utils')
octave.eval("options.DisplayWin = 0")
p, out = octave.eval("VBA_groupBMC(L, options)", nout=2)
# Save to plot
file = open(opj(outpath, 'erps_olsmean_VBAmodelcomp.pkl'), "wb")
pickle.dump(out, file)


# ########################################################################
# Mass univariate regression
###########################################################################

# Load model data
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
    goodtrials = np.where(df['badtrial'] == 0)[0]

    # Get external data for this part
    df = df.iloc[goodtrials]

    epo = epo[goodtrials]

    # Standardize data before regression
    scale = Scaler(scalings='mean')  # Says mean but is z score, see docs
    epo_z = mne.EpochsArray(scale.fit_transform(epo.get_data()),
                            epo.info)

    betasnp = []
    for idx, regvar in enumerate(regvars):
        # Standardize data
        df[regvar + '_z'] = scipy.stats.zscore(df[regvar])

        epo.metadata = df.assign(Intercept=1)  # Add an intercept for later

        # Perform regression
        names = ["Intercept"] + [regvar + '_z']
        res = mne.stats.linear_regression(epo_z, epo.metadata[names],
                                          names=names)

        # Collect betas
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

# _________________________________________________________________
# Second level test on betas

# Find channel adjacency
connect, names = mne.channels.find_ch_adjacency(epo.info,
                                                'eeg')

# Perform test for each regressor
tvals, pvals = [], []
for idx, regvar in enumerate(regvars):
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

    # Save for each regressor in case crash/stop
    np.save(opj(outpath, 'ols_2ndlevel_tval_' + regvar + '.npy'), tvals[-1])
    np.save(opj(outpath, 'ols_2ndlevel_pval_' + regvar + '.npy'), pvals[-1])

# Stack and save
tvals = np.stack(tvals)
pvals = np.stack(pvals)

np.save(opj(outpath, 'ols_2ndlevel_tvals.npy'), tvals)
np.save(opj(outpath, 'ols_2ndlevel_pvals.npy'), pvals)
np.save(opj(outpath, 'ols_2ndlevel_betas.npy'), allbetas)
all_epos.save(opj(outpath, 'ols_2ndlevel_allepochs-epo.fif'),
              overwrite=True)
np.save(opj(outpath, 'ols_2ndlevel_betasavg.npy'), beta_gavg)
