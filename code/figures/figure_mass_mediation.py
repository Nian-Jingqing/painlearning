# #########################################################################
# ERP analyses for Zoey's conditioning task
# @MP Coll, 2018, michelpcoll@gmail.com
# ERPs to shocks and relationship with behaviour
##############################################################################

import mne
from os.path import join as opj
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.io import loadmat
from bids import BIDSLayout


###############################
# Parameters
###############################
layout = BIDSLayout('/data/source')
part = ['sub-' + s for s in layout.get_subject()]

# Remove stupid pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

# Outpath for analysis
outpath = '/data/derivatives/statistics/multi_mediation'
# Outpath for figures
outfigpath = '/data/derivatives/figures/multi_mediation'
if not os.path.exists(outfigpath):
    os.mkdir(outfigpath)

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
         'usefdr': True,
         # Font sizez in plot
         'titlefontsize': 28,
         'labelfontsize': 28,
         'ticksfontsize': 26,
         'legendfontsize': 24,
         }

part = [p for p in part if p not in param['excluded']]
# Despine
plt.rc("axes.spines", top=False, right=False)
plt.rcParams['font.family'] = 'Arial Narrow'

# Load a mock data set to get times and channels pos
# Mock  info
epo = mne.read_evokeds(opj('/data/derivatives',  part[0], 'eeg',
                           part[0] + '_task-fearcond_CS+_ave.fif'))[0]

epo = epo.resample(param['testresampfreq'])


# Make a nice plot
# Add ERP line plots

meds = [[['sa1hat', 'rating', 'eegdat'],
         ['Irr. uncertainty', 'Pain rating', 'EEG']],
        [['sa1hat', 'nfrnorm', 'eegdat'], ['Irr. uncertainty', 'NFR', 'EEG']],
        [['eegdat', 'rating', 'sa1hat'], ['EEG', 'Pain rating',
                                          'Irr. uncertainty']],
        [['eegdat', 'nfrnorm', 'sa1hat'], ['EEG', 'NFR', 'Irr. uncertainty']],
        [['vhat', 'rating', 'eegdat'], ['Expectation', 'Pain rating', 'EEG']]]

# Use bootstraps or t-test against 0
testbetas = False
for med in meds:

    X, Y, M = med[0][0], med[0][1], med[0][2]
    X_lab, Y_lab, M_lab = med[1][0], med[1][1], med[1][2]
    # Load mediation output
    file = opj(outpath, 'MassMediation_X_' + X + '_Y_'
               + Y + '_M_' + M + '_nboots0.mat')
    medat = loadmat(file)

    # Path order is a, b, c', c, ab
    paths = ['a', 'b', "c'", "c", 'ab']
    pathdesc = [X_lab + ' -> ' + M_lab,
                M_lab + ' -> ' + Y_lab,
                X_lab + ' -> ' + Y_lab,
                X_lab + " -> " + Y_lab,
                X_lab + " * " + Y_lab,
                ]
    allplots = []

    for b in [0, 1, 4, 2]:  # Loop effects and skip c

        # If no bootstrap t-test against 0 using subject data
        pval = np.asarray(medat['pvals'][:, :, b])
        # pval = mne.stats.fdr_correction(pval, alpha=param['alpha'])[1]
        meanbetas = np.asarray(medat['betas'][:, :, b])
        sembetas = np.asarray(medat['betas_se'][:, :, b])
        fig = plt.figure(figsize=(12, 9))

        # Plot descritive topo data
        plot_times = [-0.1, 0.1, 0.300, 0.500, 0.8]
        times_pos = [np.abs(epo.times - t).argmin() for t in plot_times]

        topo_axis = []
        for j in [0, 2, 4, 6, 8]:
            topo_axis.append(plt.subplot2grid((4, 11),
                                              (0, j),
                                              colspan=2,
                                              rowspan=1))

        for tidx, timepos in enumerate(times_pos):
            mask = pval[:, timepos] < param['alpha']
            im, _ = mne.viz.plot_topomap(meanbetas[:, timepos],
                                         pos=epo.info,
                                         mask=mask,
                                         mask_params=dict(marker='o',
                                                          markerfacecolor='w',
                                                          markeredgecolor='k',
                                                          linewidth=0,
                                                          markersize=3),
                                         cmap='viridis',
                                         show=False,
                                         vmin=-0.1,
                                         vmax=0.1,
                                         axes=topo_axis[tidx],
                                         sensors=True,
                                         contours=0,)
            topo_axis[tidx].set_title(str(int(plot_times[tidx]*1000)) + ' ms',
                                      fontdict={'size':
                                                param['labelfontsize']})
            if tidx == 0:
                topo_axis[tidx].set_ylabel('', fontdict={'size': 15})

        cax = fig.add_axes([0.85, 0.77, 0.015, 0.15], label="cbar1")
        cbar1 = fig.colorbar(im, cax=cax,
                             orientation='vertical', aspect=10,
                             ticks=[-0.1, 0, 0.1])
        cbar1.ax.tick_params(labelsize=param['ticksfontsize'])

        # Add ERP line plots
        chan_to_plot = ['POz']

        pick = epo.ch_names.index(chan_to_plot[0])
        clrs = sns.color_palette("deep", 5)

        # Calculate standard error for shading
        sub_avg = []

        # Get mean and standard error
        mean = meanbetas[pick, :]
        sem = sembetas[pick, :]

        line_axis = []
        line_axis.append(plt.subplot2grid((4, 11),
                                          (1, 0),
                                          colspan=11,
                                          rowspan=3,))

        line_axis[0].plot(epo.times*1000, mean, label='betas',
                          linewidth=3)

        line_axis[0].fill_between(epo.times*1000,
                                  mean-sem, mean+sem, alpha=0.3,
                                  facecolor=clrs[0])
        line_axis[0].set_ylabel('Betas (path ' + paths[b] + ')',
                                fontdict={'size': param['labelfontsize']})
        line_axis[0].set_xlabel('Time (ms)',
                                fontdict={'size': param['labelfontsize']})

        line_axis[0].tick_params(labelsize=param['ticksfontsize'])

        ymin = -0.005
        ymax = 0

        # Add p values
        for tidx2, t2 in enumerate(epo.times*1000):
            if pval[pick, tidx2] < 0.05:
                line_axis[0].fill_between([t2,
                                           t2+(1000/param['testresampfreq'])],
                                          -0.005, 0, alpha=0.3,
                                          facecolor='red')

        # Append to list to combine later
        plt.suptitle('Path ' + paths[b] + ' - ' + pathdesc[b],
                     fontsize=param['titlefontsize'], y=1.05)
        plt.savefig(opj(outfigpath,
                        'ERP_mediation_results_nboots'
                        + str(chan_to_plot) + '_'
                        + paths[b] + '.svg'),
                    bbox_inches='tight',
                    pad_inches=0.02, dpi=600)
