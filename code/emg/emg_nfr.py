# #########################################################################
# EMG analyses for Zoey's conditioning task
# @MP Coll, 2020, michelpcoll@gmail.com
# #########################################################################

import pandas as pd
import numpy as np
import os
from os.path import join as opj
from bids import BIDSLayout
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns

###############################
# Parameters
###############################
layout = BIDSLayout('/data/source')
part = ['sub-' + s for s in layout.get_subject()]

# Remove stupid pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

# Parameters
param = {
         # Epoch length (in samples)
         'epochlen': 200,
         # Epoch boundaries for AUC measure
         'latencyauc': [90, 180]
         }

###########################################################################
# Load data, plot and measure AUC
###########################################################################
fig_global, ax = plt.subplots(nrows=6, ncols=6, figsize=(20, 16))
axis_rat = ax.flatten()


for idx, p in enumerate(part):
    nfrout = pd.DataFrame(index=range(54))

    outpath = opj('/data/derivatives', p, 'emg')
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    # Load physio data
    physdat = [f for f in os.listdir(opj('/data/source/', p, 'eeg'))
               if 'physio' in f][0]
    physdat = pd.read_csv(opj('/data/source', p, 'eeg', physdat), sep='\t')
    physdat.columns

    # Get ratings
    events = pd.read_csv(opj('/data/source', p, 'eeg',
                             [f for f in os.listdir(opj('/data/source/',
                                                        p, 'eeg'))
                              if 'events' in f][0]), sep='\t')

    ratings = np.asarray(events['painrating'].dropna())

    # Find shocks triggers
    trig_ons = np.where(physdat['events'] == 'shock')[0]

    # Create epochs
    epochs = []
    for t in trig_ons:
        epochs.append(np.asarray(physdat['rmsemg'])[t:t + param['epochlen']])

    epochs = np.stack(epochs)

    # Get AUC and plot all epochs
    fig, ax = plt.subplots(nrows=7, ncols=8, figsize=(20, 16))
    ax = ax.flatten()
    nfr_auc = []
    for i in range(epochs.shape[0]):
        # Get AUC
        nfr_auc.append(np.trapz(y=epochs[i,
                                         param['latencyauc'][0]:
                                             param['latencyauc'][1]]))
        # Plot
        ax[i].plot(epochs[i, :])
        ax[i].set_title('Shock ' + str(i))
        ax[i].set_xlabel('Time from trigger (ms)')

    fig.tight_layout()
    fig.savefig(opj(outpath, p + '_rmsemg_plot.png'), dpi=600)

    # Get AUC measure
    nfr_auc_z = zscore(nfr_auc)
    ratings_z = zscore(ratings)
    nfrout['nfr_auc_z'] = nfr_auc_z
    nfrout['nfr_auc'] = nfr_auc
    nfrout['ratings_z'] = ratings_z
    nfrout['ratings'] = ratings

    # Plot correlation between rating and nfr
    sns.regplot(nfr_auc_z, ratings_z, ax=axis_rat[idx])
    axis_rat[idx].set_xlabel('Z scored NFR (AUC of RMS EMG 90-180 ms)')
    axis_rat[idx].set_ylabel('Z scored pain rating')
    axis_rat[idx].set_title(p)

    nfrout.to_csv(opj(outpath, p + '_task-fearcond_nfrauc.csv'))

fig_global.tight_layout()
fig_global.savefig(opj('/data/derivatives/figures',
                       'nfr_rating_correlation.png'), dpi=600)
