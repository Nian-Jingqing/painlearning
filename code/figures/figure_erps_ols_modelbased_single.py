#-*- coding: utf-8  -*-
"""
Author: michel-pierre.coll
Date: 2020-07
Description: Plot results of stats_erps_ols_modelbased.py
"""

import mne
import pandas as pd
import numpy as np
from os.path import join as opj
import matplotlib.pyplot as plt
from bids import BIDSLayout
from mne.viz import plot_topomap
import seaborn as sns
import os
import scipy.stats
import pickle


###################################################################
# Parameters
###################################################################

layout = BIDSLayout('/data/source')
part = ['sub-' + s for s in layout.get_subject()]

# Remove stupid pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

# Outpaths for analysis
outpath = '/data/derivatives/statistics/erps_modelbased_ols_single'

outfigpath = '/data/derivatives/figures/erps_modelbased_ols_single'
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
    # Downsample to this frequency prior to analysis
    'testresampfreq': 256,
    # Excluded parts
    'excluded': ['sub-24', 'sub-31', 'sub-35', 'sub-51']
}

# exclude
part = [p for p in part if p not in param['excluded']]

# Despine
plt.rc("axes.spines", top=False, right=False)
plt.rcParams['font.family'] = 'Arial Narrow'


###################################################################
# Multivariate regression plot
###################################################################

tvals = np.load(opj(outpath, 'ols_2ndlevel_tvals.npy'))
pvals = np.load(opj(outpath, 'ols_2ndlevel_pvals.npy'))
all_epos = mne.read_epochs(opj(outpath, 'ols_2ndlevel_allepochs-epo.fif'))

beta_gavg = np.load(opj(outpath, 'ols_2ndlevel_betasavg.npy'),
                    allow_pickle=True)
allbetas = np.load(opj(outpath, 'ols_2ndlevel_betas.npy'),
                   allow_pickle=True)

regvars = ['vhat', 'sa1hat', 'sa2hat']
regvarsnames = ['Expectation', 'Irr. uncertainty', 'Est. uncertainty']


# ## Plot
# Plot descritive topo data
plot_times = [0.4, 0.6, 0.8]
times_pos = [np.abs(beta_gavg[0].times - t).argmin() for t in plot_times]

chan_to_plot = ['POz']
regvarsnamestopo = ['Expectation', 'Irr.\nuncertainty', 'Est.\nuncertainty']

for ridx, regvar in enumerate(regvars):
    fig = plt.figure(figsize=(7, 7))

    regvarname = regvarsnamestopo[ridx]

    topo_axis = []
    for j in np.arange(0, len(plot_times)*2, 2):
        topo_axis.append(plt.subplot2grid((4, 7),
                                          (0, j),
                                          colspan=2,
                                          rowspan=1))

    beta_gavg_nomast = beta_gavg[ridx].copy().drop_channels(['M1', 'M2'])
    chankeep =  [True if c not in ['M1', 'M2'] else False for c in
                 beta_gavg[ridx].ch_names]

    for tidx, timepos in enumerate(times_pos):
        im, _ = plot_topomap(beta_gavg_nomast.data[:, timepos],
                             pos=beta_gavg_nomast.info,
                             mask=pvals[ridx][timepos, chankeep] < param['alpha'],
                             mask_params=dict(marker='o',
                                              markerfacecolor='w',
                                              markeredgecolor='k',
                                              linewidth=0,
                                              markersize=4),
                             cmap='viridis',
                             show=False,
                             ch_type='eeg',
                             outlines='head',
                             extrapolate='head',
                             vmin=-0.15,
                             vmax=0.15,
                             axes=topo_axis[tidx],
                             sensors=True,
                             contours=0,)
        topo_axis[tidx].set_title(str(int(plot_times[tidx] * 1000)) + ' ms',
                                  fontdict={'size': param['labelfontsize']})
        # if tidx == 0:
        #     topo_axis[tidx].set_ylabel(regvarname,
        #                                fontdict={'size':
        #                                          param['labelfontsize']})

    cax = fig.add_axes([0.91, 0.75, 0.015, 0.15], label="cbar1")
    cbar1 = fig.colorbar(im, cax=cax,
                         orientation='vertical', aspect=10,
                         ticks=[-0.15, 0, 0.15])
    cbar1.ax.tick_params(labelsize=param['ticksfontsize'] - 6)
    cbar1.set_label('Beta', rotation=-90,
                    labelpad=20, fontdict={'fontsize': param["labelfontsize"]-5})
    fig.tight_layout()
    fig.savefig(opj(outfigpath, 'fig_ols_erps_betas_topo_' + regvar + '.svg'),
                dpi=600, bbox_inches='tight')


for c in chan_to_plot:
    for ridx, regvar in enumerate(regvars):
        fig, line_axis = plt.subplots(1, 1, figsize=(7, 7))
        regvarname = regvarsnames[ridx]
        all_epos.metadata.reset_index()
        nbins = 4
        all_epos.metadata['bin'] = 0
        all_epos.metadata['bin'], bins = pd.cut(all_epos.metadata[regvar],
                                                nbins,
                                                labels=False, retbins=True)
        all_epos.metadata['bin' + '_' + regvar] = all_epos.metadata['bin']
        # Bin labels
        bin_labels = []
        for bidx, b in enumerate(bins):
            if b < 0:
                b = 0
            if bidx < len(bins)-1:
                lab = [str(round(b, 2)) + '-'
                       + str(round(bins[bidx+1], 2))][0]
                count = np.where(all_epos.metadata['bin'] == bidx)[0].shape[0]


                # lab = lab + ' N = ' + str(count)
                bin_labels.append(lab)

        colors = {str(val): val for val in all_epos.metadata['bin'].unique()}
        evokeds = {val: all_epos['bin' + " == " + val].average()
                   for val in colors}

        pick = beta_gavg[ridx].ch_names.index(c)

        line_axis.set_ylabel('Beta (' + regvarname + ')',
                             fontdict={'size': param['labelfontsize']})
        # line_axis[0].tick_params(labelsize=12)

        for idx, bin in enumerate([str(i) for i in range(nbins)]):
            line_axis.plot(all_epos[0].times * 1000,
                           evokeds[bin].data[pick, :] * 1000000,
                           label=str(idx + 1),
                           linewidth=3,
                           color=plt.cm.get_cmap('viridis',
                                                 nbins)(idx / nbins))

        line_axis.legend(frameon=False,
                         loc='upper left',
                         fontsize=param['legendfontsize'] - 6,
                         labels = bin_labels)
        # Make it nice
        # line_axis[ridx].set_ylim((-1, 15))

        line_axis.tick_params(labelsize=12)
        line_axis.set_xlabel('Time (ms)',
                             fontdict={'size': param['labelfontsize']})
        line_axis.set_ylabel('Amplitude (uV)',
                             fontdict={'size': param['labelfontsize']})
        line_axis.axhline(0, linestyle=':', color='k')
        line_axis.axvline(0, 5, linestyle=':', color='k')
        line_axis.get_xaxis().tick_bottom()
        line_axis.get_yaxis().tick_left()
        line_axis.tick_params(labelsize=param['ticksfontsize'])
        # line_axis.set_title(regvarname, fontsize=param['titlefontsize'])
        fig.tight_layout()
        fig.savefig(opj(outfigpath,
                        'fig_ols_erps_amp_bins_' + regvar + '_'
                        + c + '.svg'),
                    dpi=600, bbox_inches='tight')

for c in chan_to_plot:
    for ridx, regvar in enumerate(regvars):
        fig, line_axis = plt.subplots(1, 1, figsize=(7, 7))

        regvarname = regvarsnames[ridx]
        all_epos.metadata.reset_index()
        pick = beta_gavg[ridx].ch_names.index(c)

        sub_avg = []
        for s in range(allbetas.shape[0]):
            sub_avg.append(allbetas[s, ridx, pick, :])
        sub_avg = np.stack(sub_avg)

        sem = scipy.stats.sem(sub_avg, axis=0)
        mean = beta_gavg[ridx].data[pick, :]

        clrs = sns.color_palette("deep", 5)

        line_axis.set_ylabel('Beta (' + regvarname + ')',
                             fontdict={'size': param['labelfontsize']})
        line_axis.set_xlabel('Time (ms)',
                             fontdict={'size': param['labelfontsize']})
        # line_axis[0].tick_params(labelsize=12)

        line_axis.plot(all_epos[0].times * 1000,
                       beta_gavg[ridx].data[pick, :],
                       label=str(idx + 1),
                       linewidth=3)
        line_axis.fill_between(all_epos[0].times * 1000,
                               mean - sem, mean + sem, alpha=0.3,
                               facecolor=clrs[0])
        # Make it nice
        line_axis.set_ylim((-0.1, 0.3))

        line_axis.axhline(0, linestyle=':', color='k')
        line_axis.axvline(0, linestyle=':', color='k')
        line_axis.get_xaxis().tick_bottom()
        line_axis.get_yaxis().tick_left()
        line_axis.tick_params(axis='both',
                              labelsize=param['ticksfontsize'])
        # line_axis.set_title(regvarname, fontsize=param['titlefontsize'])

        pvals[ridx][:, pick]
        timestep = 1024 / param['testresampfreq']
        for tidx2, t2 in enumerate(all_epos[0].times * 1000):
            if pvals[ridx][tidx2, pick] < param['alpha']:
                line_axis.fill_between([t2,
                                        t2 + timestep],
                                       -0.02, -0.005, alpha=0.3,
                                       facecolor='red')
        fig.tight_layout()
        fig.savefig(opj(outfigpath,
                        'fig_ols_erps_betas_' + regvar + '_'
                        + c + '.svg'),
                    dpi=600, bbox_inches='tight')


###################################################################
# Plot model comparison
###################################################################
file = open(opj(outpath, 'erps_olsmean_VBAmodelcomp.pkl'), "rb")
out = pickle.load(file)

modnames = ['Expectation', 'Irr.\nuncertainty', 'Est.\nuncertainty']

ep = out['ep'][0]
ef = [out['Ef'][0][0]*100,
      out['Ef'][1][0]*100,
      out['Ef'][2][0]*100]

fig, host = plt.subplots(figsize=(7, 7))

par1 = host.twinx()
color1 = '#004c6d'
color2 = '#5886a5'

x = np.arange(0.5, (len(ep))*0.75, 0.75)
x2 = [c + 0.25 for c in x]
p1 = host.bar(x, ep, width=0.25, color=color1, linewidth=1, edgecolor='k')
p2 = par1.bar(x2, ef, width=0.25, color=color2, linewidth=1, edgecolor='k')

host.set_ylim(0, 1)
par1.set_ylim(0, 100)


# host.set_xlabel("Distance")
host.set_ylabel("Exceedance probability", fontsize=param["labelfontsize"])
par1.set_ylabel("Model Frequency (%)",  fontsize=param["labelfontsize"])


for ax in [par1]:
    ax.set_frame_on(True)
    ax.patch.set_visible(False)

    plt.setp(ax.spines.values(), visible=False)
    ax.spines["right"].set_visible(True)

host.yaxis.label.set_color(color1)
par1.yaxis.label.set_color(color2)

host.spines["left"].set_edgecolor(color1)
par1.spines["right"].set_edgecolor(color2)

host.set_xticks([i+0.125 for i in x])
host.set_xticklabels(modnames, size=param['ticksfontsize'])

host.tick_params(axis='x', labelsize=param['labelfontsize'])

host.tick_params(axis='y', colors=color1, labelsize=param['labelfontsize'])
par1.tick_params(axis='y', colors=color2, labelsize=param['labelfontsize'])
fig.tight_layout()

fig.savefig(opj(outfigpath, 'erps_modelcomp.svg'), dpi=600)