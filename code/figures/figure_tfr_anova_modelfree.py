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

        pdat.append(np.float32(data[cond][-1].data))

    anova_data.append(np.stack(pdat))
    gavg[cond] = mne.grand_average(data[cond])


anova_data = np.stack(anova_data)

# # Take difference of interest for each part
diff_data = np.empty((2,) + anova_data.shape[1:])
for s in range(anova_data.shape[1]):
    diff_data[0, s, ::] = (anova_data[0, s, :] - anova_data[1, s, :])/anova_data[0, s, :]*100
    diff_data[1, s, ::] = (anova_data[2, s, :] - anova_data[3, s, :])/anova_data[1, s, :]*100

diff_data = np.squeeze(diff_data)

# Always use  chan x freq x time

pvals = np.load(opj(outpath, 'cues4_tfr_anova_pvals.npy'))
Fvals = np.load(opj(outpath, 'cues4_tfr_anova_Fvals.npy'))
p_plot = data[cond][0].copy()
p_plot.data = pvals
p_plot.data = np.where(p_plot.data < param['alpha'], 1, 0)

# ###########################################################################
# Make plot
###############################################################################


# Helper functions
def boxplot_freqs(foi, chan, time, gavg, data_all, ax, pal):
    # Colour palette for plotting
    c = 'CS-1'
    fidx = np.arange(np.where(gavg[c].freqs == foi[0])[0],
                     np.where(gavg[c].freqs == foi[1])[0])

    times = gavg[c].times
    tidx = np.arange(np.argmin(np.abs(times - time[0])),
                     np.argmin(np.abs(times - time[1])))
    cidx = gavg[c].ch_names.index(chan)

    plt_dat = data_all[:, :, cidx, :, :]
    plt_dat = plt_dat[:, :, fidx, :]
    plt_dat = plt_dat[:, :, :, tidx]
    plt_dat = np.average(plt_dat, 3)
    plt_dat.shape
    plt_dat = np.average(plt_dat, 2)

    plt_dat = pd.DataFrame(data=np.swapaxes(plt_dat, 1, 0),
                           columns=['CS-1', 'CS-2', 'CS-E', 'CS+'])
    plt_dat = pd.melt(plt_dat, var_name='Condition', value_name='Power')

    pt.half_violinplot(x='Condition', y="Power", data=plt_dat, inner=None,
                       jitter=True, color=".7", lwidth=0, width=0.6,
                       offset=0.17, cut=1, ax=ax,
                       linewidth=1, alpha=0.6, palette=pal, zorder=19)
    sns.stripplot(x='Condition', y="Power", data=plt_dat,
                  jitter=0.08, ax=ax,
                  linewidth=1, alpha=0.6, palette=pal, zorder=1)
    sns.boxplot(x='Condition', y="Power", data=plt_dat,
                palette=pal, whis=np.inf, linewidth=1, ax=ax,
                width=0.1, boxprops={"zorder": 10, 'alpha': 0.5},
                whiskerprops={'zorder': 10, 'alpha': 1},
                medianprops={'zorder': 11, 'alpha': 0.5})

    return ax


def topo_freqs(p_plotin, foi, chan, time, gavg, dcond, ax, pal, vmin=-6,
               vmax=6):
    # Colour palette for plotting
    c = 'CS-1'


    return ax


# First line are the TF plots at cz
for chan in ['Pz', 'POz', 'Cz', 'CPz', 'Fz']:
    fig = plt.figure(figsize=(18, 9))

    axes = []
    for i in [0, 2]:
        for j in [0, 2]:
            axes.append(plt.subplot2grid((4, 7),
                                         (i, j),
                                         colspan=2,
                                         rowspan=2))
    # Statistics
    axes.append(plt.subplot2grid((4, 7),
                                 (0, 5),
                                 colspan=2,
                                 rowspan=2))

    for idx, c in enumerate(conditions):
        pltdat = gavg[c]
        pick = pltdat.ch_names.index(chan)
        ch_mask = np.asarray([1 if c == chan else 0 for c in pltdat.ch_names])

        pltdat.plot(picks=[pick],
                    tmin=-0.5, tmax=1,
                    show=False,
                    cmap='viridis',
                    # vmin=-0.15,
                    # vmax=0.15,
                    title='',
                    axes=axes[idx],
                    colorbar=False,
                    )

        if idx < 2:
            axes[idx].set_xlabel('',
                                 fontdict={'fontsize': param['labelfontsize']})
            axes[idx].set_xticks([])
        else:
            axes[idx].set_xlabel('Time (s)',
                                 fontdict={'fontsize': param['labelfontsize']})
            axes[idx].tick_params(axis="x",
                                  labelsize=param['ticksfontsize'])
        if idx == 0:
            axes[idx].set_ylabel('',
                                 fontdict={'fontsize': param['labelfontsize']})
        else:
            axes[idx].set_ylabel(None,
                                 fontdict={'fontsize': param['labelfontsize']})

        if idx in [1, 3]:
            axes[idx].set_yticks([])
        else:
            axes[idx].tick_params(axis="y", labelsize=param['ticksfontsize'])

    for idx, c in enumerate(conditions):
        axes[idx].set_title(c, fontdict={"fontsize": param['titlefontsize']})

    # Pvalue plot
    p_plot.plot(picks=[pick],
                tmin=-0.2, tmax=1,
                show=False,
                cmap='Greys',
                vmin=0.1,
                vmax=1.1,
                title='',
                axes=axes[len(conditions)],
                colorbar=False,
                )
    plt.tight_layout()
    axes[-1].set_ylabel(None,
                        fontdict={'fontsize': param['labelfontsize']})
    axes[-1].set_xlabel('Time (s)',
                        fontdict={'fontsize': param['labelfontsize']})

    pos = axes[-1].get_position()
    pos.y0 = pos.y0 - 0.23      # for example 0.2, choose your value
    pos.y1 = pos.y1 - 0.23
    pos.x0 = pos.x0 - 0.06     # for example 0.2, choose your value
    pos.x1 = pos.x1 - 0.06
    axes[-1].set_position(pos)

    axes[-1].set_title("FDR corrected at " + chan,
                       fontdict={"fontsize": param['titlefontsize']})
    axes[-1].tick_params(axis="both", labelsize=param['ticksfontsize'])

    fig.text(-0.01, 0.44, 'Frequency (Hz)',
             fontdict={'fontsize': param['labelfontsize'],
                       'fontweight': 'normal'}, rotation=90)

    cax = fig.add_axes([0.58, 0.40, 0.01, 0.30],
                       label="cbar1")
    cbar1 = fig.colorbar(axes[0].images[0], cax=cax,
                         orientation='vertical', aspect=10)
    cbar1.set_label('Power (log ratio)', rotation=-90,
                    labelpad=16, fontdict={'fontsize': param['labelfontsize']})
    cbar1.ax.tick_params(labelsize=param['ticksfontsize'])

    plt.savefig(opj(outfigpath, 'TF_plots_' + chan + '.svg'),
                bbox_inches='tight', dpi=600)

    # TOPO and bar plots
    plt.close('all')
    fig2 = plt.figure(figsize=(12, 6))

    axes = []

    for i in [0, 1]:
        for j in [0, 1]:
            axes.append(plt.subplot2grid((2, 4),
                                         (i, j),
                                         colspan=1,
                                         rowspan=1))

    for j in [0]:
        axes.append(plt.subplot2grid((2, 4),
                                     (j, 2),
                                     colspan=2,
                                     rowspan=2))

    foi = [20, 21]
    time = [0.6, 1]
    data_all = np.stack(anova_data)
    boxplot_freqs(foi, chan, time, gavg, data_all, axes[-1], param['palette'])
    axes[-1].set_xlabel('Condition',
                        fontdict={'fontsize': param['labelfontsize']})
    axes[-1].set_ylabel('Power (log ratio)',
                        fontdict={'fontsize': param['labelfontsize']})
    axes[-1].tick_params(axis="both", labelsize=param['ticksfontsize'])
    # foi = [4, 6]
    # chan = 'Pz'
    # time = [0.2, 1]
    # boxplot_freqs(foi, chan, time, gavg, data_all, axes[-2], pal)

    foi = [20, 21]
    time = [0.6, 1]

    fidx = np.arange(np.where(gavg[c].freqs == foi[0])[0],
                     np.where(gavg[c].freqs == foi[1])[0])

    times = gavg[c].times
    tidx = np.arange(np.argmin(np.abs(times - time[0])),
                     np.argmin(np.abs(times - time[1])))

    p_dat = p_plot.data
    p_dat = p_dat[:, fidx, :]
    p_dat = p_dat[:, :, tidx]
    p_dat = np.average(p_dat, 2)
    p_dat = np.average(p_dat, 1)
    mask = np.where(p_dat > 0, 1, 0)

    for idx, c in enumerate(conditions):

        dcond = data_all[idx, :]
        plt_dat = dcond[:, :, fidx, :]
        plt_dat = plt_dat[:, :, :, tidx]
        plt_dat = np.average(plt_dat, 3)
        plt_dat = np.average(plt_dat, 2)

        plot_topomap(np.average(plt_dat, 0),
                     pltdat.info,
                     show=False,
                     cmap='viridis',
                     # vmin=-0.15,
                     # vmax=0.15,
                     mask=mask,
                     axes=axes[idx],
                     contours=False)

        axes[idx].set_title(c, fontdict={'fontsize': param['titlefontsize']})
    # Get data of interest

    # Second line Significance at Fz, Cz, Pz
    # Same with topo between sig freqs
    # Bar plot 20-40 Hz; 500-600 ms
    # Topo plot 20-40 Hz 500-600 ms
    plt.tight_layout()

    cax = fig2.add_axes([0.18, 0.52, 0.1, 0.05], label="cbar1")
    cbar1 = fig2.colorbar(axes[0].images[0], cax=cax,
                          orientation='horizontal', aspect=20)
    cbar1.set_label('Power (log ratio)', rotation=0,
                    labelpad=10,
                    fontdict={'fontsize': param['labelfontsize']-5})
    cbar1.ax.tick_params(labelsize=param['ticksfontsize'])

    plt.savefig(opj(outfigpath, 'TF_topobar_' + chan + '.svg'),
                bbox_inches='tight', dpi=600)


pvals = np.load(opj(outpath, 'cuesdiff_tfr_ttest_pvals.npy'))
# pvals = np.swapaxes(pvals, 0, 2)

# Same thing but for difference
for chan in ['Pz', 'POz', 'Cz', 'CPz', 'Fz']:

    conditions = ['CS-1 vs CS-2', 'CS+ vs CS-E']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    p_plot = data[cond][0].copy()
    p_plot.data = pvals
    p_plot.data = np.where(p_plot.data < param['alpha'], 1, 0)

    pow_plot = []
    pow_plot.append(data[cond][0].copy())
    pow_plot[0].data = np.mean(diff_data[0, ::], axis=0)
    pow_plot.append(data[cond][0].copy())
    pow_plot[1].data = np.mean(diff_data[1, ::], axis=0)

    for idx, c in enumerate(conditions):
        pltdat = pow_plot[idx]
        pick = pltdat.ch_names.index(chan)
        ch_mask = np.asarray([1 if c == chan else 0 for c in pltdat.ch_names])

        pltdat.plot(picks=[pick],
                    tmin=-0.5, tmax=1,
                    show=False,
                    cmap='viridis',
                    # vmin=-0.15,
                    # vmax=0.15,
                    title='',
                    axes=axes[idx],
                    colorbar=False,
                    )

        axes[idx].tick_params(axis="x", labelsize=param['ticksfontsize'])

        if idx == 0:
            axes[idx].set_ylabel('Frequency (Hz)',
                                 fontdict={'fontsize': param['labelfontsize']})
            axes[idx].set_xlabel(None,
                                 fontdict={'fontsize': param['labelfontsize']})
        else:
            axes[idx].set_ylabel(None,
                                 fontdict={'fontsize': param['labelfontsize']})
            axes[idx].set_xlabel('Time (s)',
                                 fontdict={'fontsize': param['labelfontsize']})

        if idx in [1]:
            axes[idx].set_yticks([])
        else:
            axes[idx].tick_params(axis="y", labelsize=param['ticksfontsize'])

    for idx, c in enumerate(conditions):
        axes[idx].set_title(c, fontdict={"fontsize": param['titlefontsize']})

    # Pvalue plot
    p_plot.plot(picks=[pick],
                tmin=-0.2, tmax=1,
                show=False,
                cmap='Greys',
                # vmin=0.1,
                # vmax=1.1,
                title='',
                axes=axes[len(conditions)],
                colorbar=False
                )
    plt.tight_layout()
    axes[-1].set_ylabel(None,
                        fontdict={'fontsize': param['labelfontsize']})
    axes[-1].set_xlabel('Time (s)',
                        fontdict={'fontsize': param['labelfontsize']})

    pos = axes[-1].get_position()
    pos.y0 = pos.y0
    pos.y1 = pos.y1
    pos.x0 = pos.x0 + 0.1
    pos.x1 = pos.x1 + 0.1
    axes[-1].set_position(pos)

    axes[-1].set_title("FDR corrected at " + chan,
                       fontdict={"fontsize": param['titlefontsize']})
    axes[-1].tick_params(axis="both", labelsize=param['ticksfontsize'])

    cax = fig.add_axes([0.68, 0.40, 0.01, 0.30],
                       label="cbar1")
    cbar1 = fig.colorbar(axes[0].images[0], cax=cax,
                         orientation='vertical', aspect=10)
    cbar1.set_label('Power (log ratio)', rotation=-90,
                    labelpad=18, fontdict={'fontsize': param['labelfontsize']})
    cbar1.ax.tick_params(labelsize=param['ticksfontsize'])

    plt.savefig(opj(outfigpath, 'TF_plots_diff_' + chan + '.svg'),
                bbox_inches='tight', dpi=600)

    fig2, axes = plt.subplots(1, 3, figsize=(12, 4))

    foi = [20, 21]
    time = [0.6, 1]

    fidx = np.arange(np.where(gavg['CS-1'].freqs == foi[0])[0],
                     np.where(gavg['CS-1'].freqs == foi[1])[0])

    times = gavg['CS-1'].times
    tidx = np.arange(np.argmin(np.abs(times - time[0])),
                     np.argmin(np.abs(times - time[1])))
    cidx = gavg['CS-1'].ch_names.index(chan)
    diff_data.shape
    plt_dat = diff_data[:, :, cidx, fidx, tidx]
    plt_dat = np.average(plt_dat, 2)
    plt_dat = pd.DataFrame(data=np.swapaxes(plt_dat, 1, 0),
                           columns=conditions)
    plt_dat = pd.melt(plt_dat, var_name='Condition', value_name='Power')

    pt.half_violinplot(x='Condition', y="Power", data=plt_dat, inner=None,
                       jitter=True, color=".7", lwidth=0, width=0.6,
                       offset=0.17, cut=1, ax=axes[-1],
                       linewidth=1, alpha=0.6,
                       palette=[param['palette'][0], param['palette'][3]],
                       zorder=19)
    sns.stripplot(x='Condition', y="Power", data=plt_dat,
                  jitter=0.08, ax=axes[-1],
                  linewidth=1, alpha=0.6,
                  palette=[param['palette'][0], param['palette'][3]], zorder=1)
    sns.boxplot(x='Condition', y="Power", data=plt_dat,
                palette=[param['palette'][0], param['palette'][3]],
                whis=np.inf, linewidth=1,
                ax=axes[-1],
                width=0.1, boxprops={"zorder": 10, 'alpha': 0.5},
                whiskerprops={'zorder': 10, 'alpha': 1},
                medianprops={'zorder': 11, 'alpha': 0.5})
    axes[-1].set_xlabel('',
                        fontdict={'fontsize': param['labelfontsize']})
    axes[-1].set_ylabel('Power (log ratio)',
                        fontdict={'fontsize': param['labelfontsize']})
    axes[-1].tick_params(axis="both", labelsize=param['ticksfontsize'])

    for idx, c in enumerate(conditions):
        dcond = diff_data[idx, :]
        fidx = np.arange(np.where(gavg['CS-1'].freqs == foi[0])[0],
                         np.where(gavg['CS-1'].freqs == foi[1])[0])

        times = gavg['CS-1'].times
        tidx = np.arange(np.argmin(np.abs(times - time[0])),
                         np.argmin(np.abs(times - time[1])))
        plt_dat = dcond[:, :, fidx, tidx]
        plt_dat.shape
        plt_dat = np.average(plt_dat, 2)

        p_dat = pvals
        p_dat = p_dat[:, fidx, tidx]
        p_dat = np.average(p_dat, 1)

        mask = np.where(p_dat < param['alpha'], 1, 0)

        plot_topomap(np.average(plt_dat, 0),
                     pltdat.info,
                     show=False,
                     cmap='viridis',
                     # vmaxm=0.15,
                     # vmin=-0.15,
                     mask=mask,
                     axes=axes[idx],
                     contours=False)

        axes[idx].set_title(c, fontdict={'fontsize': param['titlefontsize']})
    # Get data of interest

    # Second line Significance at Fz, Cz, Pz
    # Same with topo between sig freqs
    # Bar plot 20-40 Hz; 500-600 ms
    # Topo plot 20-40 Hz 500-600 ms
    plt.tight_layout()

    cax = fig2.add_axes([0.27, 0.40, 0.1, 0.05], label="cbar1")
    cbar1 = fig2.colorbar(axes[0].images[0], cax=cax,
                          orientation='horizontal', aspect=20)
    cbar1.set_label('Power (log ratio)', rotation=0,
                    labelpad=10,
                    fontdict={'fontsize': param['labelfontsize']-5})
    cbar1.ax.tick_params(labelsize=param['ticksfontsize'])

    plt.savefig(opj(outfigpath, 'TF_topobar_' + chan + '.svg'),
                bbox_inches='tight', dpi=600)
