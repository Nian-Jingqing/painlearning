from os.path import join as opj
import os
import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from bids import BIDSLayout
import ptitprince as pt
from scipy.io import loadmat
###############################
# Parameters
###############################
layout = BIDSLayout('/data/source')
part = ['sub-' + s for s in layout.get_subject()]

# Remove stupid pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

# Outpath for analysis
outpath = '/data/derivatives/compmodels'
# Outpath for figures
outfigpath = '/data/derivatives/figures/compmodels'
if not os.path.exists(outfigpath):
    os.mkdir(outfigpath)

param = {
         # Font sizez in plot
         'titlefontsize': 24,
         'labelfontsize': 24,
         'ticksfontsize': 22,
         'legendfontsize': 20,
         }

# Despine
plt.rc("axes.spines", top=False, right=False)
plt.rcParams['font.family'] = 'Arial Narrow'

#  ################################################################
# Figure X SCR plot
#################################################################
# A) SCR raw/cond/Block
# B) SCR predicted/cond/block
# C) Trialwise SCR pred vs actual
# D) Trialwise expected vs actual probability

# Winning model
win = 'HGF2_intercue'

# Load data
data = pd.read_csv(opj(outpath, win,  win + '_data.csv'))

# Remove shocks
data_ns = data.copy()
data_ns = data[data['cond'] != 'CS++']

# Get average SCR/cond/block
data_avg_all = data_ns.groupby(['cond_plot',
                                'block'])['scr',
                                          'pred'].mean().reset_index()

# Get SD
data_se_all = data_ns.groupby(['cond_plot',
                               'block'])['scr',
                                         'pred'].std().reset_index()
# Divide by sqrt(n)
data_se_all.scr = data_se_all.scr / np.sqrt(len(set(data_ns['sub'])))
data_se_all.pred = data_se_all.pred / np.sqrt(len(set(data_ns['sub'])))


# Init figure
fig, ax = plt.subplots(figsize=(8, 5))

off = 0.1  # Dots offset to avoid overlap

for cond in data_avg_all['cond_plot']:
    if cond[0:3] == 'CS+':
        label = 'CS+ / CSE'
        marker = 'o'
        color = "#C44E52"
        linestyle = '-'
        condoff = 0.05
    else:
        label = 'CS-1 / CS-2'
        marker = '^'
        color = '#4C72B0'
        linestyle = '--'
        condoff = -0.025
    dat_plot = data_avg_all[data_avg_all.cond_plot == cond].reset_index()
    dat_plot_se = data_se_all[data_se_all.cond_plot == cond]

    # len(dat_plot)
    if len(dat_plot) > 1:
        ax.errorbar(x=[dat_plot.block[0] + off, dat_plot.block[1] + condoff],
                    y=dat_plot.scr,
                    yerr=dat_plot_se.scr, label=label,
                    marker=marker, color=color, ecolor=color,
                    linestyle=linestyle, markersize=8, linewidth=2)
    else:
        ax.errorbar(x=[dat_plot.block[0] - off],
                    y=dat_plot.scr,
                    yerr=dat_plot_se.scr, label=label,
                    marker=marker, color=color, ecolor=color,
                    linestyle=linestyle, markersize=8, linewidth=2)

for line in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]:
    ax.axvline(x=line, linestyle=':', color='k', alpha=0.5)
ax.set_ylabel('SCR (beta estimate)', fontsize=param['labelfontsize'])
ax.set_xlabel('Block', fontsize=param['labelfontsize'])
# ax1[0].set_ylim([0.1, 0.26])
ax.tick_params(labelsize=param['ticksfontsize'])
handles, labels = ax.get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(),
          loc='best', fontsize=param["legendfontsize"], frameon=False)

fig.tight_layout()
fig.savefig(opj(outfigpath, 'scr_average.svg'), dpi=600, bbox_inches='tight')


fig, ax = plt.subplots(figsize=(8, 5))

# SAME WITH PRED
# Init figure
for cond in data_avg_all['cond_plot']:
    if cond[0:3] == 'CS+':
        label = 'CS+ / CSE'
        marker = 'o'
        color = "#C44E52"
        linestyle = '-'
        condoff = 0.05
    else:
        label = 'CS-1 / CS-2'
        marker = '^'
        color = '#4C72B0'
        linestyle = '--'
        condoff = -0.025
    dat_plot = data_avg_all[data_avg_all.cond_plot == cond].reset_index()
    dat_plot_se = data_se_all[data_se_all.cond_plot == cond]

    # len(dat_plot)
    if len(dat_plot) > 1:
        ax.errorbar(x=[dat_plot.block[0] + off, dat_plot.block[1] + condoff],
                    y=dat_plot.pred,
                    yerr=dat_plot_se.pred, label=label,
                    marker=marker, color=color, ecolor=color,
                    linestyle=linestyle, markersize=8, linewidth=2)
    else:
        ax.errorbar(x=[dat_plot.block[0] - off],
                    y=dat_plot.pred,
                    yerr=dat_plot_se.pred, label=label,
                    marker=marker, color=color, ecolor=color,
                    linestyle=linestyle, markersize=8, linewidth=2)
for line in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]:
    ax.axvline(x=line, linestyle=':', color='k', alpha=0.5)
ax.set_ylabel('Predicted SCR', fontsize=param['labelfontsize'])
ax.set_xlabel('Block', fontsize=param['labelfontsize'])
# ax1[1].set_ylim([0.1, 0.26])
ax.tick_params(labelsize=param['ticksfontsize'])

fig.tight_layout()
fig.savefig(opj(outfigpath, 'pred_scr_average.svg'), dpi=600,
            bbox_inches='tight')


fig, ax = plt.subplots(figsize=(8, 5))

# Actual vs predicted /trial
deep_pal = sns.color_palette('deep')

data_ns['cond2'] = 0
data_ns['cond2'] = np.where(data_ns['cond'] == 'CS++',
                            "CS+", data_ns['cond2'])
data_ns['cond2'] = np.where(data_ns['cond'] == 'CS-1',
                            'CS-1', data_ns['cond2'])
data_ns['cond2'] = np.where(data_ns['cond'] == 'CS-2',
                            'CS-2', data_ns['cond2'])
data_ns['cond2'] = np.where(data_ns['cond'] == 'CS+',
                            "CS+", data_ns['cond2'])

data_ns['cond2'] = np.where(data_ns['cond'] == 'CS-E',
                            "CS-E", data_ns['cond2'])

data_avg_all = data_ns.groupby(['block',
                                'trial_within_wb',
                                'cond'])['scr', 'pred',
                                         'vhat'].mean().reset_index()


ax.scatter(x=data_avg_all.trial_within_wb[data_avg_all.cond == 'CS-1'],
           y=data_avg_all.scr[data_avg_all.cond == 'CS-1'],
           facecolors='none',
           color='#4C72B0',
           alpha=1,
           label='CS-1')
ax.scatter(x=data_avg_all.trial_within_wb[data_avg_all.cond == 'CS-2'],
           y=data_avg_all.scr[data_avg_all.cond == 'CS-2'],
           facecolors='none',
           color='#0d264f',
           alpha=1,
           label='CS-2')

ax.scatter(x=data_avg_all.trial_within_wb[data_avg_all.cond == 'CS+'],
           y=data_avg_all.scr[data_avg_all.cond == 'CS+'],
           label='CS+',
           facecolors='none',
           color="#C44E52",
           alpha=1)
ax.scatter(x=data_avg_all.trial_within_wb[data_avg_all.cond == 'CS-E'],
           y=data_avg_all.scr[data_avg_all.cond == 'CS-E'],
           label='CS-E',
           facecolors='none',
           color="#55A868",
           alpha=1)
ax.scatter(x=data_avg_all.trial_within_wb[data_avg_all.cond == 'CS-1'],
           y=data_avg_all.pred[data_avg_all.cond == 'CS-1'],
           color='#4C72B0',
           alpha=0.8,
           label='CS-1')
ax.scatter(x=data_avg_all.trial_within_wb[data_avg_all.cond == 'CS-2'],
           y=data_avg_all.pred[data_avg_all.cond == 'CS-2'],
           color='#0d264f',
           alpha=0.8,
           label='CS-2')
ax.scatter(x=data_avg_all.trial_within_wb[data_avg_all.cond == 'CS+'],
           y=data_avg_all.pred[data_avg_all.cond == 'CS+'],
           color="#C44E52",
           alpha=0.8,
           label='CS+')
ax.scatter(x=data_avg_all.trial_within_wb[data_avg_all.cond == 'CS-E'],
           y=data_avg_all.pred[data_avg_all.cond == 'CS-E'],
           color="#55A868",
           alpha=0.8,
           label='CS-E')


# Find trials where new block begins
lines = []
for idx in range((len(data_avg_all.block) - 1)):
    if data_avg_all.block[idx + 1] != data_avg_all.block[idx]:
        lines.append(data_avg_all.trial_within_wb[idx] + 0.5)

for line in lines:
    ax.axvline(x=line, linestyle=':', color='k', alpha=0.5)


ax.set_ylabel('Actual / Predicted SCR', fontsize=param['labelfontsize'])
ax.set_xlabel('Trials within condition and block',
              fontsize=param['labelfontsize'])

ax.tick_params(labelsize=param['ticksfontsize'])
handles, labels = ax.get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(),
          loc='upper left', fontsize=param['legendfontsize']-6, frameon=True)

fig.tight_layout()
fig.savefig(opj(outfigpath, 'pred_scr_bytrial.svg'), dpi=600,
            bbox_inches='tight')


# Estimated quantities throught time
data_ns['cond2'] = 0
data_ns['cond2'] = np.where(data_ns['cond'] == 'CS++',
                            "CS+", data_ns['cond'])


data_avg_all = data_ns.groupby(['block',
                                'trial_within_wb_wcs',
                                'cond_plot2',
                                'cond2'])['scr',
                                          'pred',
                                          'sa1hat',
                                          'sa2hat',
                                          'vhat'].mean().reset_index()

xlabels = [r'Expected value $(\hat{\mu}_1)$',
           r'Irreducible uncertainty $(\hat{\sigma}_1)$',
           r'Estimation uncertainty $(\hat{\sigma}_2)$']
for ucue in data_avg_all['cond_plot2'].unique():
    selected = data_avg_all[data_avg_all.cond_plot2 == ucue].reset_index()

for idx, to_plot in enumerate(['vhat', 'sa1hat', 'sa2hat']):
    fig, ax = plt.subplots(figsize=(8, 5))
    for ucue in data_avg_all['cond_plot2'].unique():
        selected = data_avg_all[data_avg_all.cond_plot2 == ucue].reset_index()

        if selected.cond_plot2.loc[0][0:3] == 'CS-':
            color1 = '#4C72B0'
            color2 = '#0d264f'
            leg1 = 'CS-1'
            leg2 = 'CS-2'
        else:
            color1 = '#c44e52'
            color2 = '#55a868'
            leg1 = 'CS+'
            leg2 = 'CS-E'

        sns.lineplot(x=selected.trial_within_wb_wcs,
                     y=selected[to_plot],
                     color=color1,
                     alpha=1,
                     ax=ax,
                     label=leg1)

        if selected.block.unique().shape[0] > 1:

            selected2 = selected[selected.block
                                 == selected.block.unique()[1]]
            sns.lineplot(x=selected2.trial_within_wb_wcs,
                         y=selected2[to_plot],
                         color=color2,
                         alpha=1,
                         ax=ax,
                         label=leg2)

    ax.set_ylabel(xlabels[idx], fontsize=param['labelfontsize'])
    ax.set_xlabel('Trials', fontsize=param['labelfontsize'])

    ax.tick_params(labelsize=param['ticksfontsize'])
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc='best', fontsize=param["legendfontsize"]-6, frameon=True)

    for line in lines:
        ax.axvline(x=line, linestyle=':', color='k', alpha=0.5)

    fig.tight_layout()
    fig.savefig(opj(outfigpath, 'traj_bytrial_' + to_plot + '.svg'), dpi=600,
                bbox_inches='tight')


# ################################################################
# Parameters plot
##################################################################

fig, ax = plt.subplots(1, 4, figsize=(8, 5))
pal = sns.color_palette("deep", 5)
labels = [r'$\omega_2$', r'$\beta_0$', r'$\beta_1$', r'$\zeta$']
for idx, var in enumerate(['om_2', 'be0', 'be1', 'ze']):

    data_param = data.groupby(['sub'])[var].mean().reset_index()

    dplot = data_param.melt(['sub'])

    pt.half_violinplot(x='variable', y="value", data=dplot, inner=None,
                       jitter=True, color=pal[idx], lwidth=0, width=0.6,
                       offset=0.17, cut=1, ax=ax[idx],
                       linewidth=1, alpha=0.6, zorder=19)
    sns.stripplot(x='variable', y="value", data=dplot,
                  jitter=0.08, ax=ax[idx],
                  linewidth=1, alpha=0.6, color=pal[idx], zorder=1)
    sns.boxplot(x='variable', y="value", data=dplot,
                color=pal[idx], whis=np.inf, linewidth=1, ax=ax[idx],
                width=0.1, boxprops={"zorder": 10, 'alpha': 0.5},
                whiskerprops={'zorder': 10, 'alpha': 1},
                medianprops={'zorder': 11, 'alpha': 0.5})
    ax[idx].set_xticklabels([labels[idx]], fontsize=param['labelfontsize'])
    if idx == 0:
        ax[idx].set_ylabel('Value', fontsize=param['labelfontsize'])
    else:
        ax[idx].set_ylabel('')
    ax[idx].set_xlabel('')
    ax[idx].tick_params('y', labelsize=param['ticksfontsize']-4)
    ax[idx].tick_params('x', labelsize=param['ticksfontsize'])

    fig.tight_layout()
    fig.savefig(opj(outfigpath, 'model_parameters.svg'), dpi=600)


# ################################################################
# Model comparison plots
##################################################################

# Compare families
famcomp = loadmat(opj('/data/derivatives/compmodels/',
                      'compare_families_VBA_model_comp.mat'))

modnames = [str(m[0]) for m in famcomp['out']['options'][0][0][0][0][0][0]]
modnames = [m.replace('_nointercue', '\ncue specific') for m in modnames]
modnames = [m.replace('_intercue', '\ninter-cue') for m in modnames]
modnames.append('Family\ncue specific')
modnames.append('Family\ninter-cue')


ep = list(famcomp['out']['ep'][0][0][0])
ef = [float(ef)*100 for ef in famcomp['out']['Ef'][0][0]]

ef_fam = famcomp['out']['families'][0][0][0][0][4]
ep_fam = famcomp['out']['families'][0][0][0][0][6]

ep.append(ep_fam[0][0])
ep.append(ep_fam[0][1])
ef.append(float(ef_fam[0])*100)
ef.append(float(ef_fam[1])*100)

modnames = np.asarray(modnames)[np.asarray([0, 2, 4, 6, 1, 3, 5, 7, 8, 9])]
ep = np.asarray(ep)[np.asarray([0, 2, 4, 6, 1, 3, 5, 7, 8, 9])]
ef = np.asarray(ef)[np.asarray([0, 2, 4, 6, 1, 3, 5, 7, 8, 9])]
fig, host = plt.subplots(figsize=(12, 5))

par1 = host.twinx()
color1 = '#7293cb'
color2 = '#e1974c'

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
host.axvline(3.25, linestyle='--', color='gray')
host.axvline(6.25, linestyle='--', color='gray')

host.set_xticks([i+0.125 for i in x])
host.set_xticklabels(modnames, size=param['ticksfontsize'])

host.tick_params(axis='x', labelsize=param['labelfontsize']-10)

host.tick_params(axis='y', colors=color1, labelsize=param['labelfontsize'])
par1.tick_params(axis='y', colors=color2, labelsize=param['labelfontsize'])
fig.tight_layout()
fig.savefig(opj(outfigpath, 'model_comparison_families.svg'), dpi=600)


# Compare intercues
famcomp = loadmat(opj('/data/derivatives/compmodels/',
                      'compare_intercues_VBA_model_comp.mat'))

modnames = [str(m[0]) for m in famcomp['out']['options'][0][0][0][0][0][0]]
modnames = [m.replace('_intercue', '') for m in modnames]


ep = famcomp['out']['ep'][0][0][0]
ef = [float(ef)*100 for ef in famcomp['out']['Ef'][0][0]]

fig, host = plt.subplots(figsize=(8, 5))

par1 = host.twinx()
color1 = '#7293cb'
color2 = '#e1974c'

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
fig.savefig(opj(outfigpath, 'model_comparison_intercues.svg'), dpi=600)
