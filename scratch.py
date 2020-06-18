# %% #########################################################################
# Time-frequency analyses for Zoey's conditioning task
# @MP Coll, 2019, michelpcoll@gmail.com
##############################################################################

from mne.report import Report
import pprint
import mne
import os
from os.path import join as opj
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from mne.time_frequency import tfr_morlet
from bids import BIDSLayout

###############################
# Parameters
###############################

layout = BIDSLayout('/data/source')
part = ['sub-' + s for s in layout.get_subject()]

# Remove stupid pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

param = {
         # Additional LP filter for ERPs
         # Filter to use
         'filtertype': 'iir',
         # Length of epochs
         'erpbaseline': -0.2,
         'erpepochend': 1,
         'tfrepochstart': -1.5,
         'tfrepochend': 1.5,
         # Threshold to reject trials
         'tfr_reject': dict(eeg=500e-6),
         # TFR parameters
         'ttfreqs': np.arange(4, 60, 1),
         'freqmax': 60,
         'n_cycles': 6,
         'tfr_baseline_time': (-0.2, 0),
         'tfr_baseline_mode': 'logratio',
         'testresampfreq': 256,
         # Njobs to run the TFR
         'njobs': 20
          }

##############################################################################
# EPOCH AND TF transform
##############################################################################

for p in part:
    # ______________________________________________________
    # Make out dir
    outdir = opj('/data/derivatives',  p, 'eeg')

    strials = mne.time_frequency.read_tfrs(opj(outdir,  p + '_task-fearcond_'
                                           + 'shocks_epochs-tfr.h5'))[0]

    strials.apply_baseline(mode=param['tfr_baseline_mode'],
                           baseline=param['tfr_baseline_time'])

    # Save as npy for matlab
    np.save(opj(outdir,  p + '_task-fearcond_'
                + 'shocks_epochs-tfr.npy'), strials.data)
