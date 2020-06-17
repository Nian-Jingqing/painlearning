# #########################################################################
# ERP analyses for Zoey's conditioning task
# @MP Coll, 2018, michelpcoll@gmail.com
###########################################################################

import mne
import scipy.stats
import pandas as pd
import numpy as np
import os
from os.path import join as opj
from collections import OrderedDict
import matplotlib.pyplot as plt
from bids import BIDSLayout

###############################
# Parameters
##############################
layout = BIDSLayout('/data/source')
part = ['sub-' + s for s in layout.get_subject()]

# Remove stupid pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

# Outpath for analysis
outpath = '/data/derivatives/statistics/erps_modelfree_anova'
# Outpath for figures
outfigpath = '/data/derivatives/figures'

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
         'testresampfreq': 512,
         # Excluded parts
         'excluded': ['sub-24', 'sub-31', 'sub-35', 'sub-51']
         }

# exclude
part = [p for p in part if p not in param['excluded']]

# Despine
plt.rc("axes.spines", top=False, right=False)
plt.rcParams['font.family'] = 'Arial Narrow'

###############################
# Load data
##############################

# Epoched data
conditions = ['CS-1', 'CS-2', 'CS+', 'CS-E']
conditions_diff = ['CS-1 vs CS-2', 'CS+ vs CS-E']


data = dict()
gavg = dict()
for cond in conditions:
    data[cond] = list()
    for p in part:
        outdir = opj('/data/derivatives',  p, 'eeg')
        dat = mne.read_evokeds(opj(outdir, p + '_task-fearcond_' + cond
                                   + '_ave.fif'))[0]

        # Resample if needed
        if dat.info['sfreq'] != param['testresampfreq']:
            dat = dat.resample(param['testresampfreq'])

        data[cond].append(dat)
    gavg[cond] = mne.grand_average(data[cond])

# Get difference for each part
data_diff = dict()
gavg_diff = dict()

for condd in conditions_diff:
    data_diff[condd] = list()
    for idx, p in enumerate(part):
        dat = data['CS-1'][idx].copy()
        if condd == 'CS-1 vs CS-2':
            dat.data = data['CS-1'][idx].data - data['CS-2'][idx].data
        else:
            dat.data = data['CS+'][idx].data - data['CS-E'][idx].data

        data_diff[condd].append(dat)

    gavg_diff[condd] = mne.grand_average(data_diff[condd])

# Anova data
anovapvals = np.load(opj(outpath, 'cues4_anova_pvals.npy'))
anovaF = np.load(opj(outpath, 'cues4_anova_Fvals.npy'))

# TTtest data
ttest_pvals = np.load(opj(outpath, 'cuesdiff_ttest_pvals.npy'))
ttest_tvals = np.load(opj(outpath, 'cuesdiff_ttest_tvals.npy'))
times = np.load(opj(outpath, 'resamp_times.npy'))

###############################
# Plot topo maps
##############################

# TO MOVE IN PARAM
alpha = 0.05
# color limits for topomaps
cue_mvzlims = [-7.5, 7.5]
cue_fzlims = [0, 30]
dif_mvzlims = [-5, 5]
dif_tzlims = [0, np.max(ttest_tvals)+1]
# Plot descritive topo data
plot_times = [0, 0.1, 0.300, 0.500, 0.8]
chan_to_plot = ['POz', 'CPz', 'Fz', 'Pz', 'Oz']
clrs = ['#4C72B0', '#0d264f', '#c44e52', '#55a868']

# Find time index
times_pos = [np.abs(times - t).argmin() for t in plot_times]


# Topo for ANOVA
# Init figure and axes
labsize = param['labelfontsize']
fig = plt.figure(figsize=(8, 8))

topo_axis = []
for i in range(0, len(conditions) + 1):
    for j in range(0, len(plot_times)):
        topo_axis.append(plt.subplot2grid((len(conditions) + 1,
                                           len(plot_times)),
                                          (i, j),
                                          colspan=1,
                                          rowspan=1))

for cidx, cond in enumerate(conditions):
    for tidx, t in enumerate(times_pos):
        plot_data = gavg[cond].data[:, t] * 1000000  # to get microvolts
        count = tidx+(cidx*5)
        im, _ = mne.viz.plot_topomap(plot_data,
                                     pos=data[cond][0].info,
                                     mask=None,
                                     # cmap=param.cmap,
                                     show=False,
                                     axes=topo_axis[count],
                                     vmin=cue_mvzlims[0],
                                     vmax=cue_mvzlims[1],
                                     sensors=True,
                                     contours=0,)
        if count < 5:
            topo_axis[count].set_title(str(int(plot_times[tidx]*1000)) + ' ms',
                                       fontdict={'size': labsize})
        if count in [0, 5, 10, 15]:
            topo_axis[count].set_ylabel(cond,
                                        fontdict={'size': labsize})


# PLot statistics topo  maps
# Find positions of times to plot in topomaps
for tidx, t in enumerate(times_pos):
    plot_data = anovaF[t, :]
    im2, _ = mne.viz.plot_topomap(plot_data,
                                  pos=data[cond][0].info,
                                  mask=anovapvals[t, :] < alpha,
                                  mask_params=dict(marker='o',
                                                   markerfacecolor='w',
                                                   markeredgecolor='k',
                                                   linewidth=0, markersize=4),
                                  cmap='viridis',
                                  axes=topo_axis[tidx + 20],
                                  show=False,
                                  vmin=cue_fzlims[0],
                                  vmax=cue_fzlims[1],
                                  sensors=True,
                                  contours=0,)
    if tidx == 0:
        topo_axis[tidx + 20].set_ylabel('ANOVA',
                                        fontdict={'size': labsize})

# Add stats color bar
cax = fig.add_axes([0.95, 0.56, 0.01, 0.22], label="cbar1")
cbar1 = fig.colorbar(im, cax=cax,
                     orientation='vertical', aspect=10)
cbar1.set_label('Amplitude (uV)', rotation=-90,
                labelpad=20, fontdict={'fontsize': param["labelfontsize"]-5})
cbar1.ax.tick_params(labelsize=14)

cax2 = fig.add_axes([0.95, 0.14, 0.01, 0.12], label="cbar1")
cbar1 = fig.colorbar(im2, cax=cax2,
                     orientation='vertical', aspect=10)
cbar1.set_label('F Ratio', rotation=-90,
                labelpad=20, fontdict={'fontsize': param["labelfontsize"]-5})
cbar1.ax.tick_params(labelsize=14)

fig.savefig(opj(outfigpath, 'fig_top_4cond.svg'),
            dpi=600, bbox_inches='tight')

# Topo for ttest
# Init figure and axes
fig = plt.figure(figsize=(8, 6))

topo_axis = []
for i in range(0, len(conditions_diff) + 1):
    for j in range(0, len(plot_times)):
        topo_axis.append(plt.subplot2grid((len(conditions_diff) + 1,
                                           len(plot_times)),
                                          (i, j),
                                          colspan=1,
                                          rowspan=1))

for cidx, cond in enumerate(conditions_diff):
    for tidx, t in enumerate(times_pos):
        plot_data = gavg_diff[cond].data[:, t] * 1000000  # to get microvolts
        count = tidx+(cidx*5)
        im, _ = mne.viz.plot_topomap(plot_data,
                                     pos=data['CS-1'][0].info,
                                     mask=None,
                                     # cmap=param.cmap,
                                     show=False,
                                     axes=topo_axis[count],
                                     vmin=dif_mvzlims[0],
                                     vmax=dif_mvzlims[1],
                                     sensors=True,
                                     contours=0,)
        if count < 5:
            topo_axis[count].set_title(str(int(plot_times[tidx]*1000)) + ' ms',
                                       fontdict={'size':
                                                 param['labelfontsize']-3})
        if count in [0, 5, 10, 15]:
            topo_axis[count].set_ylabel(cond,
                                        fontdict={'size':
                                                  param['labelfontsize']-3})

# PLot statistics topo  maps
# Find positions of times to plot in topomaps
nprev = len(conditions_diff) * len(plot_times)
for tidx, t in enumerate(times_pos):
    plot_data = ttest_tvals[t, :]

    im2, _ = mne.viz.plot_topomap(plot_data,
                                  pos=data_diff[cond][0].info,
                                  mask=ttest_pvals[t, :] < alpha,
                                  mask_params=dict(marker='o',
                                                   markerfacecolor='w',
                                                   markeredgecolor='k',
                                                   linewidth=0, markersize=4),
                                  cmap='viridis',
                                  axes=topo_axis[tidx + nprev],
                                  show=False,
                                  vmin=dif_tzlims[0],
                                  vmax=dif_tzlims[1],
                                  sensors=True,
                                  contours=0,)
    if tidx == 0:
        topo_axis[tidx + nprev].set_ylabel('T-test',
                                           fontdict={'size':
                                                     param['labelfontsize']-3})

# Add stats color bar
cax = fig.add_axes([0.95, 0.56, 0.01, 0.22], label="cbar1")
cbar1 = fig.colorbar(im, cax=cax,
                     orientation='vertical', aspect=10)
cbar1.set_label('Amplitude (uV)', rotation=-90,
                labelpad=20, fontdict={'fontsize': param["labelfontsize"]-5})
cbar1.ax.tick_params(labelsize=14)

cax2 = fig.add_axes([0.95, 0.18, 0.01, 0.12], label="cbar1")
cbar1 = fig.colorbar(im2, cax=cax2,
                     orientation='vertical', aspect=10)
cbar1.set_label('T value', rotation=-90,
                labelpad=20, fontdict={'fontsize': param["labelfontsize"]-5})
cbar1.ax.tick_params(labelsize=14)
fig.savefig(opj(outfigpath, 'fig_top_2conddiff.svg'),
            dpi=600, bbox_inches='tight')


###############################
# Plot line plots
##############################
# 4 conditions
clrs = ['#4C72B0', '#0d264f', '#c44e52', '#55a868']
for chan in chan_to_plot:
    # Add ERP line plots
    fig, line_axis = plt.subplots(figsize=(6, 6))
    pick = gavg[conditions[0]].ch_names.index(chan)
    for cidx, cond in enumerate(conditions):
        # Calculate standard error for shading
        sub_avg = []
        for s in range(len(part)):
            sub_avg.append(data[cond][s].data[pick, :])
        sub_avg = np.stack(sub_avg)
        # Get standard error
        sem = scipy.stats.sem(sub_avg, axis=0)*1000000
        mean = gavg[cond].data[pick, :]*1000000

        line_axis.plot(gavg[conditions[0]].times*1000, mean, label=cond,
                       color=clrs[cidx])
        line_axis.fill_between(gavg[conditions[0]].times*1000,
                               mean-sem, mean+sem, alpha=0.3,
                               facecolor=clrs[cidx])

    line_axis.hlines(0, xmin=line_axis.get_xlim()[0],
                     xmax=line_axis.get_xlim()[1],
                     linestyle="--",
                     colors="gray")
    line_axis.vlines(0, ymin=line_axis.get_ylim()[0]/2,
                     ymax=line_axis.get_ylim()[1]/2,
                     linestyle="--",
                     colors="gray")
    line_axis.legend(frameon=False, fontsize=param['legendfontsize'])
    # line_axis[0].set_title('Grand average ERP at ' + chan_to_plot[0],
    #                        fontdict={'size': 14})
    line_axis.set_xlabel('Time (ms)',
                         fontdict={'size': param['labelfontsize']})
    line_axis.set_ylabel('Amplitude (uV)',
                         fontdict={'size': param['labelfontsize']})

    line_axis.set_xticks(np.arange(-200, 1100, 200))
    # line_axis.set_xticklabels(np.arange(0, 900, 100))
    line_axis.tick_params(axis='both', which='major',
                          labelsize=param['ticksfontsize'],
                          length=5, width=1, direction='out', color='k')
    fig.tight_layout()
    fig.savefig(opj(outfigpath, 'fig_lineplot_4cond_' + chan + '.svg'),
                dpi=600, bbox_inches='tight')

    # Get p values
    # p_vals = np.squeeze(anovapvals[:, pick])
    #
    # ymin = -1.2
    # ymax = -1
    # for tidx, t in enumerate(gavg[conditions[0]].times*1000):
    #     if p_vals[tidx] < alpha:
    #         line_axis[0].vlines(t, ymin=ymin,
    #                             ymax=ymax,
    #                             linestyle="-",
    #                             colors="red",
    #                             alpha=0.1)
    fig.tight_layout()
    fig.savefig(opj(outfigpath, 'fig_lineplot_4cond_' + chan + '.svg'),
                dpi=600, bbox_inches='tight')

# 2 conditions
clrs = ['#55a868', '#c44e52']
for chan in chan_to_plot:
    # Add ERP line plots
    fig, line_axis = plt.subplots(figsize=(6, 6))
    pick = gavg_diff[conditions_diff[0]].ch_names.index(chan)
    for cidx, cond in enumerate(conditions_diff):
        # Calculate standard error for shading
        sub_avg = []
        for s in range(len(part)):
            sub_avg.append(data_diff[cond][s].data[pick, :])
        sub_avg = np.stack(sub_avg)
        # Get standard error
        sem = scipy.stats.sem(sub_avg, axis=0)*1000000
        mean = gavg_diff[cond].data[pick, :]*1000000

        line_axis.plot(gavg_diff[conditions_diff[0]].times*1000, mean,
                       label=cond,
                       color=clrs[cidx])
        line_axis.fill_between(gavg_diff[conditions_diff[0]].times*1000,
                               mean-sem, mean+sem, alpha=0.3,
                               facecolor=clrs[cidx])

    line_axis.hlines(0, xmin=line_axis.get_xlim()[0],
                     xmax=line_axis.get_xlim()[1],
                     linestyle="--",
                     colors="gray")
    line_axis.vlines(0, ymin=line_axis.get_ylim()[0]/2,
                     ymax=line_axis.get_ylim()[1]/2,
                     linestyle="--",
                     colors="gray")
    line_axis.legend(frameon=False, fontsize=param['legendfontsize'])
    # line_axis[0].set_title('Grand average ERP at ' + chan_to_plot[0],
    #                        fontdict={'size': 14})
    line_axis.set_xlabel('Time (ms)',
                         fontdict={'size': param['labelfontsize']})
    line_axis.set_ylabel('Amplitude (uV)',
                         fontdict={'size': param['labelfontsize']})

    line_axis.set_xticks(np.arange(-200, 1100, 200))
    # line_axis.set_xticklabels(np.arange(0, 900, 100))
    line_axis.tick_params(axis='both', which='major',
                          labelsize=param['ticksfontsize'],
                          length=5, width=1, direction='out', color='k')

    # Get p values
    p_vals = np.squeeze(ttest_pvals[:, pick])

    ymin = -1.2
    ymax = -1
    for tidx, t in enumerate(gavg[conditions[0]].times*1000):
        if p_vals[tidx] < alpha:
            line_axis.vlines(t, ymin=ymin,
                             ymax=ymax,
                             linestyle="-",
                             colors="red",
                             alpha=0.5)

    fig.tight_layout()
    fig.savefig(opj(outfigpath, 'fig_lineplot_2conddiff_' + chan + '.svg'),
                dpi=600, bbox_inches='tight')


# Single trials line plot
# Load metadata
all_meta = pd.read_csv('/data/derivatives/task-fearcond_erpsmeta.csv')

# Average across conditions
all_meta_clean = all_meta[all_meta['badtrial'] == 0]

lppcols = [c for c in list(all_meta_clean.columns.values) if 'amp_' in c]


for col in lppcols:
    fig, line_axis = plt.subplots(figsize=(8, 6))
    data_avg_all = all_meta_clean.groupby(['condblock_join',
                                           'block'])[col].mean().reset_index()


# Standard error across all conditons
# Get sd
    data_se_all = all_meta_clean.groupby(['condblock_join',
                                          'block'])[col].std().reset_index()
    npart = len(set(all_meta_clean['participant_id']))
    data_se_all[col] = data_se_all[col]/np.sqrt(npart)

    off = 0.1  # Dots offset to avoid overlap

    # Create a new df with colors, styles, etc.
    for cond in data_avg_all['condblock_join']:
        if cond[0:3] == 'CS+':
            label = 'CS+ / CSE'
            marker = 'o'
            color = '#d53e4f'
            linestyle = '-'
            condoff = 0.05
        else:
            label = 'CS-1 / CS-2'
            marker = '^'
            color = '#3288bd'
            linestyle = '--'
            condoff = -0.025
        dat_plot = data_avg_all[data_avg_all.condblock_join
                                == cond].reset_index()
        dat_plot_se = data_se_all[data_se_all.condblock_join == cond]

        if len(dat_plot) > 1:
            line_axis.errorbar(x=[dat_plot.block[0]+off,
                                  dat_plot.block[1]+condoff],
                               y=dat_plot[col],
                               yerr=dat_plot_se[col], label=label,
                               marker=marker, color=color, ecolor=color,
                               linestyle=linestyle, markersize=8, linewidth=2)
        else:
            line_axis.errorbar(x=[dat_plot.block[0]-off],
                               y=dat_plot[col],
                               yerr=dat_plot_se[col], label=label,
                               marker=marker, color=color, ecolor=color,
                               linestyle=linestyle, markersize=8, linewidth=2)
    for line in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]:
        line_axis.axvline(x=line, linestyle=':', color='k')

    line_axis.set_ylabel('Mean amplitude\n400-800 ms (Z scored)',
                         fontsize=param['labelfontsize'])
    line_axis.set_xlabel('Block',
                         fontsize=param['labelfontsize'])

    line_axis.set_xticks([1, 2, 3, 4, 5, 6, 7])
    line_axis.tick_params(labelsize=param['ticksfontsize'])
    handles, labels = line_axis.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    line_axis.legend(by_label.values(), by_label.keys(),
                     loc='best', fontsize=param["legendfontsize"]-4,
                     frameon=True)

    fig.tight_layout()
    fig.savefig(opj(outfigpath, 'figure_strials_' + col + '.svg'),
                bbox_inches='tight', dpi=600)
