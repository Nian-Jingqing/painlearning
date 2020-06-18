# #########################################################################
# ERP analyses for Zoey's conditioning task
# @MP Coll, 2020, michelpcoll@gmail.com
##############################################################################

import mne
import pandas as pd
import os
from os.path import join as opj
from bids import BIDSLayout
import numpy as np
import matplotlib.pyplot as plt
from bids import BIDSLayout
from mne.viz import plot_topomap
import seaborn as sns


###############################
# Parameters
###############################
layout = BIDSLayout('/data/source')
part = ['sub-' + s for s in layout.get_subject()]

# Remove stupid pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

# Outpath for analysis
outpath = '/data/derivatives/statistics'
if not os.path.exists(outpath):
    os.mkdir(outpath)

outpath = '/data/derivatives/statistics/erps_modelfree_anova'
if not os.path.exists(outpath):
    os.mkdir(outpath)

outfigpath = '/data/derivatives/figures/shocks_bins'
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
         'titlefontsize': 24,
         'labelfontsize': 24,
         'ticksfontsize': 22,
         'legendfontsize': 20,
         }

part = [p for p in part if p not in param['excluded']]

# Epoched data
mod_data = pd.read_csv('/data/derivatives/task-fearcond_alldata.csv')

evokeds, epochs, sub_mod_dat = [], [], []
for p in part:
    outdir = opj('/data/derivatives',  p, 'eeg')

    evokeds.append(mne.read_evokeds(opj(outdir,
                                    p + '_task-fearcond_' + 'shock'
                                    + '_ave.fif'))[0])

    epochs.append(mne.read_epochs(opj(outdir,
                                      p + '_task-fearcond_' + 'shock'
                                      + '_singletrials-epo.fif')))

    sub_mod_dat = mod_data[(mod_data['participant_id'] == p)
                           & (mod_data['cond'] == 'CS++')]

    epochs[-1].metadata = pd.concat([epochs[-1].metadata.reset_index(),
                                     sub_mod_dat.reset_index()], axis=1)

    epochs[-1] = mne.set_eeg_reference(epochs[-1], ['Cz'])[0]


all_epos = mne.concatenate_epochs(epochs)


chan_to_plot = ['POz', 'Pz', 'Oz', 'CPz', 'Cz', 'Fz']
regvars = ['vhat', 'sa1hat', 'sa2hat', 'ratings_z']
regvarsnames = ['Expectation', 'Irr. uncertainty', 'Est. uncertainty',
                'Pain ratings']


all_epos.metadata['painrating'][all_epos.metadata.participant_id == 'sub-23']

for c in chan_to_plot:
    fig, line_axis = plt.subplots(1, len(regvars), figsize=(16, 5))
    for ridx, regvar in enumerate(regvars):
        regvarname = regvarsnames[ridx]

        all_epos.metadata['bin'] = 0
        all_epos.metadata['bin'] = pd.cut(all_epos.metadata[regvar], 3,
                                          labels=False)
        colors = {str(val): val for val in all_epos.metadata['bin'].unique()}
        evokeds = {val: all_epos['bin' + " == " + val].average()
                   for val in colors}

        pick = epochs[0].ch_names.index(c)
        clrs = sns.color_palette("deep", 5)

        line_axis[ridx].set_ylabel('Beta (' + regvarname + ')',
                                   fontdict={'size': param['labelfontsize']})
        # line_axis[0].tick_params(labelsize=12)

        for idx, bin in enumerate(['0', '1', '2']):
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
                    'fig_shocks_bin_'
                    + c + '.svg'),
                dpi=600, bbox_inches='tight')
