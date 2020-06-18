# %% markdown
# # Plot

# %%
import mne
import pandas as pd
import numpy as np
from os.path import join as opj
import matplotlib.pyplot as plt
from bids import BIDSLayout
from mne.viz import plot_topomap
import seaborn as sns

# %% markdown
# ## Parameters

# %%
layout = BIDSLayout('/data/source')
part = ['sub-' + s for s in layout.get_subject()]

# Remove stupid pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

# Outpaths for analysis
outpath = '/data/derivatives/statistics/erps_modelbased_ols'

outfigpath = '/data/derivatives/figures'

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

# %% markdown
#  Load data from stats_erps_modelbasedregression.py

# %%
tvals = np.load(opj(outpath, 'ols_2ndlevel_tvals.npy'))
pvals = np.load(opj(outpath, 'ols_2ndlevel_pvals.npy'))
all_epos = mne.read_epochs(opj(outpath, 'ols_2ndlevel_allepochs-epo.fif'))
beta_gavg = np.load(opj(outpath, 'ols_2ndlevel_betasavg.npy'),
                    allow_pickle=True)
allbetas = np.load(opj(outpath, 'ols_2ndlevel_betas.npy'),
                   allow_pickle=True)

regvars = ['vhat', 'sa1hat', 'sa2hat']
regvarsnames = ['Expectation', 'Irr. uncertainty', 'Est. uncertainty']


# %% markdown
# ## Plot

# %%

# Plot descritive topo data
plot_times = [0, 0.1, 0.300, 0.500, 0.8]
times_pos = [np.abs(beta_gavg[0].times - t).argmin() for t in plot_times]

chan_to_plot = ['POz', 'Pz', 'Oz', 'CPz', 'Cz', 'Fz']

for ridx, regvar in enumerate(regvars):
    fig = plt.figure(figsize=(9, 7))

    regvarname = regvarsnames[ridx]

    topo_axis = []
    for j in [0, 2, 4, 6, 8]:
        topo_axis.append(plt.subplot2grid((4, 11),
                                          (0, j),
                                          colspan=2,
                                          rowspan=1))

    for tidx, timepos in enumerate(times_pos):
        im, _ = plot_topomap(beta_gavg[ridx].data[:, timepos],
                             pos=beta_gavg[ridx].info,
                             mask=pvals[ridx][timepos, :] < param['alpha'],
                             mask_params=dict(marker='o',
                                              markerfacecolor='w',
                                              markeredgecolor='k',
                                              linewidth=0,
                                              markersize=3),
                             cmap='viridis',
                             show=False,
                             vmin=-0.15,
                             vmax=0.15,
                             axes=topo_axis[tidx],
                             sensors=True,
                             contours=0,)
        topo_axis[tidx].set_title(str(int(plot_times[tidx]*1000)) + ' ms',
                                  fontdict={'size': param['labelfontsize']})
        if tidx == 0:
            topo_axis[tidx].set_ylabel(regvarname,
                                       fontdict={'size':
                                                 param['labelfontsize']})

    cax = fig.add_axes([0.91, 0.75, 0.015, 0.15], label="cbar1")
    cbar1 = fig.colorbar(im, cax=cax,
                         orientation='vertical', aspect=10,
                         ticks=[-0.15, 0, 0.15])
    cbar1.ax.tick_params(labelsize=param['ticksfontsize']-6)

    fig.tight_layout()
    fig.savefig(opj(outfigpath, 'fig_ols_erps_betas_topo_' + regvar + '.svg'),
                dpi=600, bbox_inches='tight')


for c in chan_to_plot:
    fig, line_axis = plt.subplots(1, 3, figsize=(10, 5))
    for ridx, regvar in enumerate(regvars):
        regvarname = regvarsnames[ridx]
        all_epos.metadata.reset_index()
        all_epos.metadata['bin'] = 0
        all_epos.metadata['bin'] = pd.cut(all_epos.metadata[regvar], 4,
                                          labels=False)
        colors = {str(val): val for val in all_epos.metadata['bin'].unique()}
        evokeds = {val: all_epos['bin' + " == " + val].average()
                   for val in colors}

        pick = beta_gavg[ridx].ch_names.index(c)
        clrs = sns.color_palette("deep", 5)

        line_axis[ridx].set_ylabel('Beta (' + regvarname + ')',
                                   fontdict={'size': param['labelfontsize']})
        # line_axis[0].tick_params(labelsize=12)

        for idx, bin in enumerate(['0', '1', '2', '3']):
            line_axis[ridx].plot(all_epos[0].times*1000,
                                 evokeds[bin].data[pick, :]*1000000,
                                 label=str(idx+1),
                                 linewidth=3,
                                 color=plt.cm.get_cmap('viridis', 5)(idx/5))

        line_axis[ridx].legend(frameon=False,
                               loc='upper left',
                               fontsize=param['legendfontsize']-5)
        # Make it nice
        # line_axis[ridx].set_ylim((-1, 15))

        line_axis[ridx].tick_params(labelsize=12)
        line_axis[ridx].set_xlabel('Time (ms)',
                                   fontdict={'size': param['labelfontsize']})
        line_axis[ridx].set_ylabel('Amplitude (uV)',
                                   fontdict={'size': param['labelfontsize']})
        line_axis[ridx].axhline(0, linestyle=':', color='k')
        line_axis[ridx].axvline(0, 5, linestyle=':', color='k')
        line_axis[ridx].get_xaxis().tick_bottom()
        line_axis[ridx].get_yaxis().tick_left()
        line_axis[ridx].tick_params(labelsize=param['ticksfontsize'])
        line_axis[ridx].set_title(regvarname, fontsize=param['titlefontsize'])

    fig.tight_layout()
    fig.savefig(opj(outfigpath,
                    'fig_ols_erps_betas_bins_' + regvar + '_'
                    + c + '.svg'),
                dpi=600, bbox_inches='tight')
