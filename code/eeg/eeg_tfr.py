##########################################################################
# Time-frequency analyses for Zoey's conditioning task
# @MP Coll, 2019, michelpcoll@gmail.com
##########################################################################

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
         # Length of epochs
         'erpbaseline': -0.2,
         'erpepochend': 1,
         'tfrepochstart': -1,
         'tfrepochend': 2,
         # Threshold to reject trials
         'tfr_reject': dict(eeg=500e-6),
         # TFR parameters
         'ttfreqs': np.arange(2, 41, 1),
         'n_cycles': np.arange(2, 41, 1)/2,
         'tfr_baseline_time': (-0.5, 0),
         'tfr_baseline_mode': 'logratio',
         'testresampfreq': 256,
         # Njobs to run the TFR
         'njobs': 20
          }

##############################################################################
# EPOCH AND TF transform
##############################################################################

removed_frame = pd.DataFrame(index=part)
removed_frame['percleft_cue'] = 999
removed_frame['percleft_shock'] = 999
percleft_cue = []
percleft_shock = []

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

    # ______________________________________________________
    # Additional filter
    raw = raw.filter(l_freq=None, h_freq=200)

    # Load trial info in scr data
    events = pd.read_csv(layout.get(subject=p[-2:], extension='tsv',
                                    suffix='events',
                                    return_type='filename')[0], sep='\t')

    # Update samples stamp of events because some files were resampled
    valid = mne.find_events(raw, verbose=False)[:, 2] < 1000
    evsamples = mne.find_events(raw,verbose=False)[:, 0][valid]
    events['sample'] = evsamples

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

    events_c['goodtrials'] = goodtrials

    # Save percent of cues removed
    percleft_cue.append(np.sum(goodtrials)/468*100)

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

    # # TFR single trials
    strials = tfr_morlet(tf_cues_strials,
                         freqs=param['ttfreqs'],
                         n_cycles=param['n_cycles'],
                         return_itc=False,
                         use_fft=True,
                         decim=int(1024/param["testresampfreq"]),
                         n_jobs=param['njobs'],
                         average=False)

    tf_cues_strials = None

    # Remove unused part
    strials.crop(tmin=param['erpbaseline'],
                 tmax=param['erpepochend'])

    strials.save(opj(outdir,  p + '_task-fearcond_'
                     + 'epochs-tfr.h5'), overwrite=True)

    # Drop bad
    strials_good = strials[np.where(np.asarray(goodtrials) == 1)[0]]

    strials = None

    # Average each condition
    induced = dict()
    for cond in events_id.keys():

        induced = strials_good[strials_good.metadata.trial_cond4
                               == cond].average()
        induced.save(opj(outdir,  p + '_task-fearcond_' + cond
                         + '_avg-tfr.h5'), overwrite=True)

        # Baseline to plot
        induced.apply_baseline(mode=param['tfr_baseline_mode'],
                               baseline=param['tfr_baseline_time'])
        # Plot in report
        chans_to_plot = ['Cz', 'Pz', 'CPz', 'Oz', 'Fz', 'FCz']
        figs_tfr = []
        for c in chans_to_plot:
            pick = induced.ch_names.index(c)
            figs_tfr.append(induced.plot(picks=[pick],
                                         tmin=param['erpbaseline'],
                                         tmax=param['erpepochend'],
                                         show=False,
                                         ))

        report.add_slider_to_section(figs_tfr,
                                     captions=chans_to_plot,
                                     section='TFR',
                                     title='Zscored TFR')

    # # same thing for shocks
    events_s = events[events.trigger_info == 'shock']
    events_s['cue_num'] = 255
    events_s['empty'] = 0
    events_shocks = np.asarray(events_s[['sample', 'empty', 'cue_num']])
    events_id = {
                 'shock': 255,
                 }
    #
    # # Reject very bad trials
    tf_shocks = mne.Epochs(
                           raw,
                           events=events_shocks,
                           event_id=events_id,
                           tmin=param['tfrepochstart'],
                           baseline=None,
                           tmax=param['tfrepochend'],
                           preload=True,
                           verbose=False,
                           reject=param['tfr_reject'],
                           reject_tmin=param['erpbaseline'],
                           reject_tmax=param['erpepochend'])

    goodtrials = [0 if len(li) > 0 else 1 for li in tf_shocks.drop_log]
    events_s['goodtrials'] = goodtrials
    percleft_shock.append(np.sum(goodtrials)/54*100)

    tf_shock_strials = mne.Epochs(
                                  raw,
                                  events=events_shocks,
                                  event_id=events_id,
                                  tmin=param['tfrepochstart'],
                                  baseline=None,
                                  metadata=events_s,
                                  tmax=param['tfrepochend'],
                                  preload=True,
                                  verbose=False)

    strials_shock = tfr_morlet(tf_shock_strials,
                               freqs=param['ttfreqs'],
                               n_cycles=param['n_cycles'],
                               return_itc=False,
                               use_fft=True,
                               decim=int(1024/param["testresampfreq"]),
                               n_jobs=param['njobs'],
                               average=False)

    # Remove unused parts
    strials_shock.crop(tmin=param['erpbaseline'],
                       tmax=param['erpepochend'])

    strials_shock.save(opj(outdir,  p + '_task-fearcond_' + 'shock_'
                           + 'epochs-tfr.h5'), overwrite=True)

    strials_shock_good = strials_shock[np.where(np.asarray(goodtrials)
                                                == 1)[0]]

    inducedshock = strials_shock_good.average()
    inducedshock.save(opj(outdir,  p + '_task-fearcond_' + 'shock'
                          + '_avg-tfr.h5'), overwrite=True)

    # Baseline to plot
    inducedshock.apply_baseline(mode=param['tfr_baseline_mode'],
                                baseline=param['tfr_baseline_time'])

    # Plot in report
    chans_to_plot = ['Cz', 'Pz', 'CPz', 'Oz', 'Fz', 'FCz']
    figs_tfr = []
    for c in chans_to_plot:
        pick = inducedshock.ch_names.index(c)
        figs_tfr.append(inducedshock.plot(picks=[pick],
                                          tmin=param['erpbaseline'],
                                          tmax=param['erpepochend'],
                                          show=False))
    report.add_slider_to_section(figs_tfr,
                                 captions=chans_to_plot,
                                 section='TFR_shock',
                                 title='TFR shocks')

    # Save report
    report.save(opj(outdir,  p + '_task-fearcond'
                    + '_tfr.html'),
                open_browser=False, overwrite=True)

    plt.close('all')

# Save rejection stats (should be the same as erps)
removed_frame['percleft_cue'] = percleft_cue
removed_frame['percleft_shock'] = percleft_shock

removed_frame.to_csv('/data/derivatives/task-fearcond_tfr_rejectionstats.csv')