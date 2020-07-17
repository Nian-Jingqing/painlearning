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
from mne.stats import spatio_temporal_cluster_1samp_test as clust_1s_ttest

###############################
# Parameters
###############################
layout = BIDSLayout('/data/source')
part = ['sub-' + s for s in layout.get_subject()]

# Remove stupid pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

# Outpath for analysis
outpath = '/data/derivatives/statistics/tfr_modelbased_ols'
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

betas, betasnp = [], []

# Loop participants and load single trials file
allbetasnp, all_epos = [], []

for p in part:

    # Get external data for this part
    df = mod_data[mod_data['sub'] == p]

    # Drop shocked trials
    df_noshock = df[df['cond'] != 'CS++']

    # Load single epochs file (cotains one epoch/trial)
    epo = read_tfrs(opj('/data/derivatives',  p, 'eeg',
                        p + '_task-fearcond_epochs-tfr.h5'))[0]

    # drop bad trials
    goodtrials = epo.metadata['goodtrials'] == 1
    epo = epo[goodtrials == 1]
    df = df_noshock.reset_index()[goodtrials == 1]

    # Baseline
    epo = epo.apply_baseline(mode='logratio',
                             baseline=(-0.2, 0))

    # Don't test baseline to reduce computational demands
    epo = epo.crop(tmin=0, tmax=1)

    # If not empty
    if len(df) != 0:

        # Vectorize, Zscore, Linear
        clf = make_pipeline(Vectorizer(),
                            StandardScaler(),
                            LinearRegression(n_jobs=param['njobs']))

        # Standardize data
        for regvar in regvars:
            df[regvar + '_z'] = ((df[regvar] - np.average(df[regvar]))
                                 / np.std(df[regvar]))

        no = [r + '_z' for r in regvars]
        # Fit regression
        clf.fit(epo.data, df[no])

        betasnp = []
        for idx, regvar in enumerate(no):
            out = np.reshape(clf['linearregression'].coef_[idx],
                             (epo.data.shape[1], epo.data.shape[2],
                              epo.data.shape[3]))
            betasnp.append(out)

        allbetasnp.append(np.stack(betasnp))


# Stack all data
allbetas = np.stack(allbetasnp)

# Create MNE EpochsArray
np.save(opj(outpath, 'ols_2ndlevel_allbetas.npy'), allbetas)
np.save(opj(outpath, 'resamp_times.npy'), epo.times)
np.save(opj(outpath, 'resamp_freqs.npy'), epo.freqs)

# #########################################################################
# Perform second level test on betas
##########################################################################

allbetas = np.load(opj(outpath, 'ols_2ndlevel_allbetas.npy'))
epo = read_tfrs(opj('/data/derivatives',  part[0], 'eeg',
                    part[0] + '_task-fearcond_epochs-tfr.h5'))[0]

# stat_fun_hat = partial(ttest_1samp_no_p, sigma=1e-3)

# Find connectivity structure
chan_connect, _ = mne.channels.find_ch_adjacency(epo.info, 'eeg')

# Cobine frequency and channel connectivityg
connectivity = mne.stats.combine_adjacency(len(epo.freqs),
                                           chan_connect)


tvals, pvals = [], []
for idx, regvar in enumerate(regvars):

    # Test each predictor
    betas_tfce = allbetas[:, idx, ::]

    # Swap time and channels to get time x freq x chan
    betas_tfce = np.swapaxes(betas_tfce, 1, 3)
    # Keep original shape
    oshape = betas_tfce.shape

    # Reshape in a sub x time x vertices frame
    betas_tfce = np.reshape(betas_tfce, (betas_tfce.shape[0],
                                         betas_tfce.shape[1],
                                         betas_tfce.shape[2]*
                                         betas_tfce.shape[3]))
    # TFCE dict
    tval, _, pval, _, = clust_1s_ttest(betas_tfce,
                                       n_permutations=param['nperms'],
                                       threshold=dict(start=0,
                                                      step=0.2),
                                       connectivity=connectivity,
                                    #    stat_fun=stat_fun_hat,
                                       max_step=1,
                                       n_jobs=param['njobs'],
                                       seed=param['random_state'])

    # Reshape back to data and append
    tvals.append(np.reshape(tval, (oshape[1],
                                   oshape[2],
                                   oshape[3])
                            ))
    pvals.append(np.reshape(pval, (oshape[1],
                                   oshape[2],
                                   oshape[3])
                            ))

    # Reshape in chan x freq x time to fit with data
    tvals[-1] = np.swapaxes(tvals[-1], 2, 0)
    pvals[-1 ]= np.swapaxes(pvals[-1], 2, 0)

    np.save(opj(outpath, 'ols_2ndlevel_tfr_pvals_' + regvar + '_.npy'),
            pvals[-1])
    np.save(opj(outpath, 'ols_2ndlevel_tfr_tvals_' + regvar + '_.npy'), tvals)


    # FDR
    # Reshape data in a single vector for t-test

    # # t-test
    # testdata = allbetas[:, idx, ::]
    # shapet = testdata.shape
    # testdata = testdata.reshape(shapet[0], shapet[1] * shapet[2] * shapet[3])
    # tval = ttest_1samp_no_p(testdata, sigma=1e-3)
    # pval = scipy.stats.t.sf(
    #     np.abs(tval), shapet[0] - 1) * 2  # two-sided pvalue

    # # FDR correction
    # _, pval = mne.stats.fdr_correction(pval, alpha=param['alpha'])

    # tvals.append(np.reshape(tval, shapet[1:]))
    # pvals.append(np.reshape(pval, shapet[1:]))

tvals = np.stack(tvals)
pvals = np.stack(pvals)

np.save(opj(outpath, 'ols_2ndlevel_tfr_pvals.npy'), pvals)
np.save(opj(outpath, 'ols_2ndlevel_tfr_tvals.npy'), tvals)

