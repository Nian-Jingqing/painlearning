# #########################################################################
# ERP analyses for Zoey's conditioning task
# @MP Coll, 2018, michelpcoll@gmail.com
###########################################################################

import mne
import pandas as pd
import numpy as np
import os
from os.path import join as opj
import matplotlib.pyplot as plt
from bids import BIDSLayout
from mne.time_frequency import read_tfrs
from scipy.stats import zscore

###############################
# Parameters
##############################
layout = BIDSLayout('/data/source')
part = ['sub-' + s for s in layout.get_subject()]

# Remove stupid pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

# Outpath for analysis
outpath = '/data/derivatives/statistics/tfr_modelbased_ols'
# Outpath for figures
outfigpath = '/data/derivatives/figures/tfr_modelbased_ols'

if not os.path.exists(outpath):
    os.mkdir(outpath)
if not os.path.exists(outfigpath):
    os.mkdir(outfigpath)

param = {
         # Njobs for permutations
         'njobs': 15,
         # New sampling rate to downsample single trials
         'resamp': 250,
         # Alpha Threshold
         'alpha': 0.05,
         # Number of permutations
         'nperms': 10000,
         # Threshold to reject trials
         'erpreject': dict(eeg=500e-6),
         # Random state to get same permutations each time
         'random_state': 23,
         # Font sizez in plot
         'titlefontsize': 24,
         'labelfontsize': 24,
         'ticksfontsize': 22,
         'legendfontsize': 20,
         # Downsample to this frequency prior to analysis
         'testresampfreq': 256,
         # Excluded parts
         'excluded': ['sub-24', 'sub-31', 'sub-35', 'sub-51'],
         # Color palette
         'palette': ['#4C72B0', '#0d264f', '#55a868', '#c44e52']

         }

# exclude
part = [p for p in part if p not in param['excluded']]

# Despine
plt.rc("axes.spines", top=False, right=False)
plt.rcParams['font.family'] = 'Arial Narrow'


###############################
# Load data
##############################
betas = np.load(opj(outpath, 'ols_2ndlevel_allbetas.npy'))

pvals = np.load(opj(outpath, 'ols_2ndlevel_tfr_pvals.npy'))
tvals = np.load(opj(outpath, 'ols_2ndlevel_tfr_tvals.npy'))

# Mock  info
epo = read_tfrs(opj('/data/derivatives',  part[0], 'eeg',
                    part[0] + '_task-fearcond_epochs-tfr.h5'))[0]

epo.apply_baseline(mode='logratio',
                              baseline=(-0.2, 0))

epo.crop(tmin=0, tmax=1, fmin=4, fmax=40)


regvarsnames = ['Expectation', 'Irr. uncertainty', 'Est. uncertainty']

# ###########################################################################
# Make plot
###############################################################################

chans_to_plot = ['Pz', 'POz', 'CPz', 'Cz', 'Fz']
for idx, regvar in enumerate(regvarsnames):

    betas_plot = np.average(betas[:, idx, ...], axis=0)
    betas_plot.shape
    betas_plot = zscore(betas_plot)

    pvals_plot = pvals[idx, ...]
    pvals_mask = np.where(pvals_plot < param['alpha'], 1, 0)

    pvals_plot = mne.time_frequency.AverageTFR(info=epo.info,
                                               data=pvals_mask,
                                               times=epo.times,
                                               freqs=epo.freqs,
                                               nave=1)

    beta_gavg_plot = mne.time_frequency.AverageTFR(info=epo.info,
                                                   data=betas_plot,
                                                   times=epo.times,
                                                   freqs=epo.freqs,
                                                   nave=1)

    for chan in chans_to_plot:

        pick = epo.ch_names.index(chan)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        beta_gavg_plot.plot(picks=[pick],
                            tmin=-0.2, tmax=1,
                            show=False,
                            cmap='viridis',
                            colorbar=False,
                            title='',
                            vmin=-2, vmax=2,
                            axes=ax[0]
                            )
        pvals_plot.plot(picks=[pick],
                        tmin=-0.2, tmax=1,
                        show=False,
                        cmap='Greys',
                        title='',
                        colorbar=False,
                        axes=ax[1],
                        vmin=0.1
                        )
        ax[0].set_title(regvar + ' - Betas at ' + chan,
                        fontsize=param['titlefontsize'])
        ax[0].set_ylabel("Frequency (Hz)",
                         fontsize=param['labelfontsize'])
        ax[1].set_xlabel('Time (s)', fontsize=param['labelfontsize'])
        ax[0].set_xlabel('Time (s)', fontsize=param['labelfontsize'])
        ax[1].set_ylabel('')

        ax[1].set_title('FDR corrected at ' + chan,
                        fontsize=param['titlefontsize'])
        ax[0].tick_params(axis="both",
                          labelsize=param['ticksfontsize'])
        ax[1].tick_params(axis="both",
                          labelsize=param['ticksfontsize'])
        cax = fig.add_axes([0.40, 1.15, 0.20, 0.02],
                           label="cbar1")
        cbar1 = fig.colorbar(ax[0].images[0], cax=cax,
                             orientation='horizontal', aspect=10)
        cbar1.set_label('Betas (Z scored)', rotation=0,
                        labelpad=7, fontdict={'fontsize':
                                              param['labelfontsize']-5})
        cbar1.ax.tick_params(labelsize=param['ticksfontsize']-5)

        fig.savefig(opj(outfigpath, 'TF_plots_oslbetas_' + regvar
                        + "_" + chan + '.svg'),
                    bbox_inches='tight', dpi=600)

    # Topo plot
    fig, (ax1, ax2) = plt.subplots(2, 6, figsize=(12, 4))
    titles = ['-200-0', '0-200', '200-400', '400-600',
              '600-800',
              '800-1000']
    for idx, times in enumerate([[-0.2, 0], [0, 0.2], [0.2, 0.4], [0.4, 0.6],
                                 [0.6, 0.8],
                                 [0.8, 1]]):

        beta_gavg_plot.plot_topomap(tmin=times[0], tmax=times[1],
                                    fmin=8, fmax=13,
                                    vmin=-2, vmax=2,
                                    cmap='viridis', axes=ax1[idx],
                                    colorbar=False, show=False,
                                    contours=False,
                                    title=titles[idx])
        beta_gavg_plot.plot_topomap(tmin=times[0], tmax=times[1], fmin=15,
                                    fmax=30,
                                    vmin=-2, vmax=2,
                                    cmap='viridis', axes=ax2[idx],
                                    contours=False,
                                    colorbar=False, show=False,
                                    title='')

        ax1[idx].set_title(titles[idx],
                           fontsize=param["labelfontsize"])

    ax1[0].set_ylabel('Alpha\n(8-13 Hz)', fontsize=param["labelfontsize"])
    ax2[0].set_ylabel('Beta\n(15-30 Hz)', fontsize=param["labelfontsize"])
    fig.savefig(opj(outfigpath, 'TF_plots_oslbetas_'
                    + regvar + '_topo.svg'), dpi=600, bbox_inches='tight')
