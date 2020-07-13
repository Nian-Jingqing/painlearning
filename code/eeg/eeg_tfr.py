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
         'erpepochend': 0.1,
         'tfrepochstart': -1,
         'tfrepochend': 2,
         # Threshold to reject trials
         'tfr_reject': dict(eeg=250e-6),
         # TFR parameters
         'ttfreqs': np.arange(2, 41, 1),
         'n_cycles': np.arange(2, 41, 1)/2,
         'tfr_baseline_time': (-0.5, 0),
         'tfr_baseline_mode': 'logratio',
         'testresampfreq': 512,
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
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # _______________________________________________________
    # Initialise MNE report
    report = Report(verbose=False, subject=p,
                    title='TFR report for part ' + p)

    report.add_htmls_to_section(htmls=pprint.pformat(param),
                                captions='Parameters',
                                section='Parameters')

    # ______________________________________________________
    # Load cleaned raw file
    raw = mne.io.read_raw_fif(opj(outdir,
                                  p + '_task-fearcond_cleanedeeg_raw.fif'),
                              preload=True)

    # Load trial info in scr data
    events = pd.read_csv(layout.get(subject=p[-2:], extension='tsv',
                                    suffix='events',
                                    return_type='filename')[0], sep='\t')

    # Update samples stamp of events because some files were resampled
    evsamples = mne.find_events(raw)[:, 0][mne.find_events(raw)[:, 2] < 1000]
    events['sample'] = evsamples

    # Remove distinction between CS-1 and CS-2
    # events['event'] = np.where(events['event'] == 97, 96, events['event'])
    # ________________________________________________________
    # Epoch according to condition

    # Drop unused channels
    chans_to_drop = [c for c in ['HEOGL', 'HEOGR', 'VEOGL',
                                 'STI 014', 'Status'] if c in raw.ch_names]
    raw.drop_channels(chans_to_drop)

    events['empty'] = 0
    events_c = events[events['trial_type'].notna()]

    # Remove shocks
    events_c = events_c[events_c['trial_type'] != 'CS+S']

    # # ________________________________________________________
    # # Epoch according to condition
    events_id = {
                 'CS-1': 1,
                 'CS-2': 2,
                 'CS+':  3,
                 'CS-E': 4,
                 }

    events_c['cue_num'] = [events_id[s] for s in events_c.trial_cond4]
    events_cues = np.asarray(events_c[['sample', 'empty', 'cue_num']])

    # Reject very bad trials
    tf_cues = mne.Epochs(
                         raw,
                         events=events_cues,
                         event_id=events_id,
                         tmin=param['tfrepochstart'],
                         baseline=None,
                         tmax=param['tfrepochend'],
                         preload=True,
                         verbose=False,
                         reject=param['tfr_reject'],
                         reject_tmin=param['erpbaseline'],
                         reject_tmax=param['erpepochend'])

    goodtrials = [0 if len(li) > 0 else 1 for li in tf_cues.drop_log]

    tf_cues_strials = mne.Epochs(
                                 raw,
                                 events=events_cues,
                                 event_id=events_id,
                                 tmin=param['tfrepochstart'],
                                 baseline=None,
                                 metadata=events_c,
                                 tmax=param['tfrepochend'],
                                 preload=True,
                                 verbose=False)
    raw = None
    # # TFR single trials
    strials = tfr_morlet(tf_cues_strials,
                         freqs=param['ttfreqs'],
                         n_cycles=param['n_cycles'],
                         return_itc=False,
                         use_fft=True,
                         decim=int(1024/param["testresampfreq"]),
                         n_jobs=param['njobs'],
                         average=False)

    # # Apply baseline
    # strials.apply_baseline(mode=param['tfr_baseline_mode'],
    #                        baseline=param['tfr_baseline_time'])
    #
    # # Remove unused parts
    # strials.crop(tmin=param['erpbaseline'],
    #              tmax=param['erpepochend'])
    tf_cues_strials = None
    strials.save(opj(outdir,  p + '_task-fearcond_'
                     + 'epochs-tfr.h5'), overwrite=True)

    # Drop bad
    strials_good = strials[np.where(np.asarray(goodtrials) == 1)[0]]
    strials = None
    induced = dict()
    for cond in events_id.keys():

        induced = strials_good[strials_good.metadata.trial_cond4
                               == cond].average()
        induced.save(opj(outdir,  p + '_task-fearcond_' + cond
                         + '_avg-tfr.h5'), overwrite=True)

        # chans_to_plot = ['Cz', 'Pz', 'CPz', 'Oz', 'Fz', 'FCz']
        # figs_tfr = []
        # for c in chans_to_plot:
        #     pick = induced.ch_names.index(c)
        #     figs_tfr.append(induced.plot(picks=[pick],
        #                                  tmin=-0.2, tmax=1,
        #                                  show=False,
        #                                  ))
        #
        # report.add_slider_to_section(figs_tfr,
        #                              captions=chans_to_plot,
        #                              section='TFR',
        #                              title='Zscored TFR')

    # # same thing for shocks
    # events_s = events[events.trigger_info == 'shock']
    # events_s['cue_num'] = 255
    # events_s['empty'] = 0
    # events_shocks = np.asarray(events_s[['sample', 'empty', 'cue_num']])
    # events_id = {
    #              'shock': 255,
    #              }
    #
    # # Reject very bad trials
    # tf_shocks = mne.Epochs(
    #                        raw,
    #                        events=events_shocks,
    #                        event_id=events_id,
    #                        tmin=param['tfrepochstart'],
    #                        baseline=None,
    #                        tmax=param['tfrepochend'],
    #                        preload=True,
    #                        verbose=False,
    #                        reject=param['tfr_reject'],
    #                        reject_tmin=param['erpbaseline'],
    #                        reject_tmax=param['erpepochend'])
    #
    # inducedshock = tfr_morlet(tf_shocks,
    #                           freqs=param['ttfreqs'],
    #                           n_cycles=param['n_cycles'],
    #                           return_itc=False,
    #                           decim=int(1024/param["testresampfreq"]),
    #                           n_jobs=param['njobs'],
    #                           average=True)
    # inducedshock.apply_baseline(mode=param['tfr_baseline_mode'],
    #                             baseline=param['tfr_baseline_time'])
    #
    # # Remove unused parts
    # inducedshock.crop(tmin=param['erpbaseline'],
    #                   tmax=param['erpepochend'])
    #
    # inducedshock.save(opj(outdir,  p + '_task-fearcond_' + 'shock'
    #                       + '_avg-tfr.h5'), overwrite=True)
    #
    # chans_to_plot = ['Cz', 'Pz', 'CPz', 'Oz', 'Fz', 'FCz']
    # figs_tfr = []
    # for c in chans_to_plot:
    #     pick = inducedshock.ch_names.index(c)
    #     figs_tfr.append(inducedshock.plot(picks=[pick],
    #                                       tmin=-0.2, tmax=1,
    #                                       show=False))
    # report.add_slider_to_section(figs_tfr,
    #                              captions=chans_to_plot,
    #                              section='TFR_shock',
    #                              title='Zscored TFR shocks')
    #
    # tf_shocks_strials = mne.Epochs(
    #                              raw,
    #                              events=events_shocks,
    #                              event_id=events_id,
    #                              tmin=param['tfrepochstart'],
    #                              baseline=None,
    #                              tmax=param['tfrepochend'],
    #                              preload=True,
    #                              verbose=False)
    #
    # # TFR single trials
    # strials = tfr_morlet(tf_shocks_strials,
    #                      freqs=param['ttfreqs'],
    #                      n_cycles=param['n_cycles'],
    #                      return_itc=False,
    #                      decim=int(1024/param["testresampfreq"]),
    #                      n_jobs=param['njobs'],
    #                      average=False)
    #
    # strials.apply_baseline(mode=param['tfr_baseline_mode'],
    #                        baseline=param['tfr_baseline_time'])
    #
    # strials.crop(tmin=param['erpbaseline'],
    #              tmax=param['erpepochend'])
    #
    # strials.save(opj(outdir,  p + '_task-fearcond_'
    #                  + 'shocks_epochs-tfr.h5'), overwrite=True)
    #
    # # Save as npy for matlab
    # np.save(opj(outdir,  p + '_task-fearcond_'
    #             + 'shocks_epochs-tfr.npy'), strials.data)
    #
    # report.save(opj(outdir,  p + '_task-fearcond'
    #                 + '_avg-tfr.html'),
    #             open_browser=False, overwrite=True)
    plt.close('all')
