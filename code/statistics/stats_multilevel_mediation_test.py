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
from scipy.io import loadmat
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
         'testresampfreq': 256,
         # excluded participants
         'excluded': ['sub-24', 'sub-31', 'sub-35', 'sub-51'],
         # Use FDR (if false witll use TFCE)
         'usefdr': True
         }

part = [p for p in part if p not in param['excluded']]


# Load model
meddat = loadmat(opj('/data/derivatives/statistics',
                     'multi_mediation',
                     'MassMediation_X_eegdat_Y_nfrnorm_M_sa1hat_nboots0.mat'))

# GEt a frame with sub x paths x channels x time
meddat['sbetas'].shape

# Test each path with a one-sample t-testdat
pvals, tvals = [], []
for idx in range(meddat['sbetas'].shape[1]):
    pathdat = meddat['sbetas'][:, idx, ...]
    testdat = pathdat.reshape(pathdat.shape[0],
                              pathdat.shape[1]*pathdat.shape[2])

    tval, pval = scipy.stats.ttest_1samp(testdat, popmean=0)
    _, pval = mne.stats.fdr_correction(pval, param['alpha'])

    pvals.append(pval.reshape(pathdat.shape[1:]))
    tvals.append(tval.reshape(pathdat.shape[1:]))

pvals = np.stack(pvals)
tvals = np.stack(tvals)

pd.Series(pvals[0, ...].flatten()).plot()

np.min(pvals[0, ...])
pd.Series(meddat['pvals'][0, ...].flatten()).plot()

a = mne.stats.fdr_correction(meddat['pvals'][0, ...], param['alpha'])
