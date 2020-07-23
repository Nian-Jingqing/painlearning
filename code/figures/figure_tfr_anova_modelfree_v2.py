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
import ptitprince as pt
import seaborn as sns
from mne.viz import plot_topomap

###############################
# Parameters
##############################
layout = BIDSLayout('/data/source')
part = ['sub-' + s for s in layout.get_subject()]

# Remove stupid pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

# Outpath for analysis
outpath = '/data/derivatives/statistics/tfr_modelfree_anova'
# Outpath for figures
outfigpath = '/data/derivatives/figures/tfr_modelfree_anova'

if not os.path.exists(outpath):
    os.mkdir(outpath)
if not os.path.exists(outfigpath):
    os.mkdir(outfigpath)

param = {
         # Alpha Threshold
         'alpha': 0.05,
         # Font sizez in plot
         'titlefontsize': 24,
         'labelfontsize': 24,
         'ticksfontsize': 22,
         'legendfontsize': 20,
         # Excluded parts
         'excluded': ['sub-24', 'sub-31', 'sub-35', 'sub-51'],
         # Color palette
         'palette': ['#4C72B0', '#0d264f', '#55a868', '#c44e52'],
         # range on colormaps
         'pwrv': [-0.2, 0.2]

         }

# exclude
part = [p for p in part if p not in param['excluded']]

# Despine
plt.rc("axes.spines", top=False, right=False)
plt.rcParams['font.family'] = 'Arial Narrow'


###############################
# Load data
##############################
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
        data[cond][-1].apply_baseline(mode='logratio',
                                      baseline=(-0.2, 0))

        data[cond][-1].crop(tmin=0, tmax=1)


        pdat.append(np.float32(data[cond][-1].data))

    anova_data.append(np.stack(pdat))
    gavg[cond] = mne.grand_average(data[cond])


anova_data = np.stack(anova_data)

# # Take difference of interest for each part
# # Take difference of interest for each part
csplusvscs1 = np.empty((1,) + anova_data.shape[1:])
csevscs2 = np.empty((1,) + anova_data.shape[1:])
csplusvscse = np.empty((1,) + anova_data.shape[1:])
csplusvscse2 = np.empty((1,) + anova_data.shape[1:])

# Calculate differences
for s in range(anova_data.shape[1]):

    csplusvscs1[0, s, ::] = (anova_data[2, s, :] - anova_data[0, s, :])
    csevscs2[0, s, ::] = (anova_data[3, s, :] - anova_data[1, s, :])
    csplusvscse[0, s, ::] = ((anova_data[2, s, :] - anova_data[3, s, :]) - (anova_data[0, s, :] - anova_data[1, s, :]))


# ###########################################################################
# Make plot
###############################################################################


# _________________________________________________________________
# Differences plot
# pvals = np.swapaxes(pvals, 0, 2)

for diff_data, savename, title, ylabel in zip([csplusvscs1, csevscs2, csplusvscse],
                               ['csplusvscs1', 'csevscs2', 'csplusvscse'],
                               ['Acqusition (CS+ vs CS-1)',
                                'Trace (CS-E vs CS-2)',
                                'Extinction (interaction)'],
                               [True, False, False]):
    pvals = np.load(opj(outpath,
                        'cuesdiff_tfr_ttest_pvals' + savename + '.npy'))
    tvals = np.load(opj(outpath,
                        'cuesdiff_tfr_ttest_tvals' + savename + '.npy'))

    # Plot difference
    for chan in ['POz']:

        fig, ax = plt.subplots(figsize=(5, 4))

        p_plot = data[cond][0].copy()
        p_plot.data = pvals

        p_plot.data = np.where(p_plot.data < param['alpha'], 1, 0)

        pltdat = data[cond][0].copy()
        pltdat.data = np.mean(diff_data[0, ::], axis=0)

        pick = pltdat.ch_names.index(chan)
        ch_mask = np.asarray([1 if c == chan else 0
                                for c in pltdat.ch_names])

        fig2 = pltdat.plot(picks=[pick],
                    tmin=-0.5, tmax=1,
                    show=False,
                    cmap='Greys',
                    vmin=param['pwrv'][0],
                    vmax=param['pwrv'][1],
                    title='',
                    axes=ax,
                    colorbar=False,
                    )

        powsig = pltdat.copy().crop(tmin=0, tmax=1)

        powsig.data = np.where(p_plot.data == 1, pltdat.data, np.nan)
        fig3 = powsig.plot(picks=[pick],
                    tmin=-0.2, tmax=1,
                    show=False,
                    cmap='viridis',
                    vmin=-0.2,
                    vmax=0.2,
                    title='',
                    axes=ax,
                    colorbar=False,
                    )

        ax.tick_params(axis="x", labelsize=param['ticksfontsize'])

        ax.set_xlabel('Time (ms)',
                                fontdict={'fontsize': param['labelfontsize']})

        ax.set_xticks(ticks=np.arange(0, 1.2, 0.2))
        ax.set_xticklabels(labels=[str(i) for i in np.arange(0, 1200, 200)])

        if ylabel:
            ax.set_ylabel('Frequency (Hz)',
                                fontdict={'fontsize': param['labelfontsize']})
        else:
            ax.set_ylabel('',
                    fontdict={'fontsize': param['labelfontsize']})

        ax.tick_params(axis="y", labelsize=param['ticksfontsize'])

        ax.set_title(title, fontdict={"fontsize": param['titlefontsize']})

        plt.tight_layout()

        plt.savefig(opj(outfigpath, 'TF_plots_diff_' + chan + '_'
                        + savename + '.svg'),
                    bbox_inches='tight', dpi=600)

    fig, axtopo = plt.subplots(figsize=(4, 4))


    time = [0.5, 1]
    foi = [20, 25]
    fidx = np.arange(np.where(gavg['CS-1'].freqs == foi[0])[0],
                    np.where(gavg['CS-1'].freqs == foi[1])[0])

    times = gavg['CS-1'].times
    tidx = np.arange(np.argmin(np.abs(times - time[0])),
                    np.argmin(np.abs(times - time[1])))
    ddata = np.mean(diff_data[0, ::], axis=0)
    plt_dat = np.average(ddata[:, :, tidx], 2)
    plt_dat = np.average(plt_dat[:, fidx], 1)

    p_dat = pvals
    p_dat = np.average(p_dat[:, :, tidx], 2)
    p_dat = np.average(p_dat[:, fidx], 1)

    mask = np.where(p_dat < param['alpha']/3, 1, 0)

    plot_topomap(plt_dat,
                pltdat.info,
                show=False,
                cmap='viridis',
                vmin=-0.2,
                vmax=0.2,
                mask_params=dict(markersize=8),
                mask=mask,
                axes=axtopo,
                contours=False)

    axtopo.set_title(title, fontdict={'fontsize': param['titlefontsize']})
    # Get data of interest

    # Second line Significance at Fz, Cz, Pz
    # Same with topo between sig freqs
    # Bar plot 20-40 Hz; 500-600 ms
    # Topo plot 20-40 Hz 500-600 ms
    plt.tight_layout()



    plt.savefig(opj(outfigpath, 'TF_diff_topo_' + chan
                    + '_' + savename + '.svg'),
                bbox_inches='tight', dpi=600)

#Generate a colorbar
fig3, cax = plt.subplots(1, 2, figsize=(0.5, 2))

cbar1 = fig2.colorbar(ax.images[2], cax=cax[1],
                    orientation='vertical', aspect=2)
cbar1.set_label('Power \n difference', rotation=-90,
                labelpad=30,
                fontdict={'fontsize': param['labelfontsize']-5})
cbar1.ax.tick_params(labelsize=param['ticksfontsize']-4)


cbar2 = fig2.colorbar(ax.images[0], cax=cax[0],
                    orientation='vertical', aspect=2)
cbar2.set_label('', rotation=-90,
                labelpad=30,
                fontdict={'fontsize': param['labelfontsize']-5})
cbar2.ax.tick_params(size=0, labelsize=0)
fig3.tight_layout()
fig3.savefig(opj(outfigpath, 'diff_colorbar.svg'), dpi=600,
             bbox_inches='tight')



###################################################################
# Plot power in each condition
###################################################################

for chan in ['POz']:


    for idx, c in enumerate(['CS-1', 'CS-2', 'CS+', 'CS-E']):
        fig, ax = plt.subplots(figsize=(5, 4))

        pltdat = gavg[c]
        pick = pltdat.ch_names.index(chan)

        pltdat.plot(picks=[pick],
                    tmin=-0.5, tmax=1,
                    show=False,
                    cmap='viridis',
                    vmin=-0.3,
                    vmax=0.3,
                    title='',
                    axes=ax,
                    colorbar=False,
                    )


        ax.set_xlabel('Time (ms)',
                                fontdict={'fontsize': param['labelfontsize']})
        ax.tick_params(axis="x",
                                labelsize=param['ticksfontsize'])
        ax.set_xticks(ticks=np.arange(0, 1.2, 0.2))
        ax.set_xticklabels(labels=[str(i) for i in np.arange(0, 1200, 200)])

        ax.set_ylabel('Frequency (Hz)',
                                fontdict={'fontsize': param['labelfontsize']})

        ax.tick_params(axis="y", labelsize=param['ticksfontsize'])

        ax.set_title(c, fontdict={"fontsize": param['titlefontsize']})

        plt.tight_layout()

        plt.savefig(opj(outfigpath, 'TF_plots_' + chan
                        + '_' + c + '.svg'),
                    bbox_inches='tight', dpi=600)
