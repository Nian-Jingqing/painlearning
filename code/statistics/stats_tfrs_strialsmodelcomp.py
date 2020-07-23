#-*- coding: utf-8  -*-
"""
Author: michel-pierre.coll
Date: 2020-07-17 15:08:26
Description: perform regression between computational estimates and LPP amp
             for each part and get BIC for model comparison.
TODO:
"""
# xhost +"local:docker@" && docker run -it --rm -e "MKL_DEBUG_CPU_TYPE=5" -v /media/mp/lxhdd/2020_painlearning:/data -v /home/mp/gdrive/projects/2020_painlearning/code:/code -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY mpcoll2/eeg2020:latest


import mne
import pandas as pd
import numpy as np
import os
from os.path import join as opj
from bids import BIDSLayout
import scipy.stats
import statsmodels.api as sm
from oct2py import octave
import statsmodels.api as sm
from mne.time_frequency import read_tfrs
import matplotlib.pyplot as plt
from scipy.stats import zscore

###############################
# Parameters
###############################
layout = BIDSLayout('/data/source')
part = ['sub-' + s for s in layout.get_subject()]

# Remove stupid pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

# Outpath for analysis
outpath = '/data/derivatives/statistics/tfrs_strialsmean_regression'
if not os.path.exists(outpath):
    os.makedirs(outpath)

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
         'titlefontsize': 24,
         'labelfontsize': 24,
         'ticksfontsize': 22,
         'legendfontsize': 20,
         }

part = [p for p in part if p not in param['excluded']]

# ########################################################################
# Extract mean power for single tirals
###########################################################################

mod_data = pd.read_csv('/data/derivatives/task-fearcond_alldata.csv')



# Loop participants and load single trials file
allbetasnp, all_epos = [], []

times = [0.5, 1]
freqs = [8, 13]
chan = ['POz']

# Loop freqbands
for freqs in [[8, 13], [15, 30]]:
    bic_tfr = pd.DataFrame(index=part, data={'vhat': 999,
                                          'sa1hat': 999,
                                          'sa2hat': 999})

    bic_beta = pd.DataFrame(index=part, data={'vhat': 999,
                                            'sa1hat': 999,
                                            'sa2hat': 999})

    # Loop variables of interest
    for variable in ['vhat', 'sa1hat', 'sa2hat']:
        # Loop for part
        for p in part:

                # Get external data for this part
            df = mod_data[mod_data['sub'] == p]

            # Drop shocked trials
            df = df[df['cond'] != 'CS++']

            # Load single epochs file (cotains one epoch/trial)
            epo = read_tfrs(opj('/data/derivatives',  p, 'eeg',
                                p + '_task-fearcond_epochs-tfr.h5'))[0]

            # Pick channel
            epo = epo.pick_channels(chan)

            # Extract data at POz in time and frequency
            dat = np.squeeze(epo.crop(tmin=times[0], tmax=times[1],
                                      fmin=freqs[0],
                            fmax=freqs[1]).data)

            # Average in time and frequency
            dat = np.average(dat, axis=2)
            dat = np.average(dat, axis=1)

            # Drop bad trials
            goodtrials = np.where(df['badtrial'] == 0)[0]
            df = df.iloc[goodtrials]
            dat = dat[goodtrials]

            # Perform the regression
            df['power'] = zscore(dat)
            df[variable] = zscore(df[variable])

            # Regression with one term + intercept
            mod = sm.OLS(df["power"],  sm.add_constant(df[variable]),
                    hasconst=True).fit()

            # Save BIC and beta
            bic_tfr.loc[p, variable] = mod.bic
            bic_beta.loc[p, variable] = mod.params[variable]



    # Use octave to run the VBA-toolbox
    octave.push('L', np.asarray(bic_tfr.transpose())*-1)
    octave.addpath('/code/VBA-toolbox-master')
    octave.addpath('/code/VBA-toolbox-master/core')
    octave.addpath('/code/VBA-toolbox-master/core/display')
    octave.addpath('/code/VBA-toolbox-master/utils')
    octave.eval("options.DisplayWin = 0")
    p, out = octave.eval("VBA_groupBMC(L, options)", nout=2)



    ###################################################################
    # Plot the model comparison results
    ###################################################################

    modnames = ['Expectation', 'Irr.\nuncertainty', 'Est.\nuncertainty']


    ep = out['ep'][0]
    ef = [out['Ef'][0][0]*100,
        out['Ef'][1][0]*100,
        out['Ef'][2][0]*100]

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

    host.tick_params(axis='x', labelsize=param['labelfontsize']-5)

    host.tick_params(axis='y', colors=color1, labelsize=param['labelfontsize'])
    par1.tick_params(axis='y', colors=color2, labelsize=param['labelfontsize'])
    fig.tight_layout()
