# #########################################################################
# ERP analyses for Zoey's conditioning task
# @MP Coll, 2018, michelpcoll@gmail.com
##############################################################################

from mne.report import Report
import pprint
import mne
import os
from os.path import join as opj
import pandas as pd
import numpy as np
from mne.viz import plot_evoked_joint as pej
from bids import BIDSLayout


###############################
# Parameters
##############################
layout = BIDSLayout('/data/source')
part = ['sub-' + s for s in layout.get_subject()]

# Remove stupid pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

param = {
         # Additional LP filter for ERPs
         'erplpfilter': 30,
         # Filter to use
         'filtertype': 'iir',
         # Length of baseline
         'erpbaseline': -0.2,
         'erpepochend': 1,
         # Threshold to reject trials
         'erpreject': dict(eeg=300e-6),
         # Threshold to reject shock trials
         'erprejectshock': dict(eeg=500e-6),
         # New sampling rate to downsample single trials
         'resamp': 256
         }

# ########################################################################
# EPOCH AND SAVE ERPS
##############################################################################

# Initialise array to collect rejection stats
reject_stats = pd.DataFrame(data={'part': part, 'perc_removed_cues': 9999,
                                  'perc_removed_shocks': 9999,
                                  'cs+': 0, 'cs-1': 0, 'cs-2': 0, 'cs-e': 0,
                                  'shocks': 0})

for p in part:

    # ______________________________________________________
    # Make out dir
    outdir = opj('/data/derivatives',  p, 'eeg')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # _______________________________________________________
    # Initialise MNE report
    report = Report(verbose=False, subject=p,
                    title='ERP report for part ' + p)

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

    # Drop unused channels
    chans_to_drop = [c for c in ['HEOGL', 'HEOGR', 'VEOGL',
                                 'STI 014', 'Status'] if c in raw.ch_names]
    raw.drop_channels(chans_to_drop)

    # Filter for erpss
    raw.filter(
        None,
        param['erplpfilter'],
        method=param['filtertype'],
        filter_length='auto',
        phase='zero',
        fir_design='firwin',
        pad="reflect_limited")

    # Add empty column to make it easier to create the event array
    events['empty'] = 0
    events_c = events[events['trial_type'].notna()]

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

    erp_cues = mne.Epochs(
                         raw,
                         events=events_cues,
                         event_id=events_id,
                         tmin=param['erpbaseline'],
                         baseline=(param['erpbaseline'], 0),
                         tmax=param['erpepochend'],
                         preload=True,
                         verbose=False,
                         reject=param['erpreject']
                         )

    reject_stats.loc[reject_stats.part == p,
                     reject_stats.columns
                     == 'perc_removed_cues'] = ((468-len(erp_cues))/468*100)
    reject_stats.loc[reject_stats.part == p,
                     reject_stats.columns == 'cs+'] = len(erp_cues['CS+'])
    reject_stats.loc[reject_stats.part == p,
                     reject_stats.columns == 'cs-2'] = len(erp_cues['CS-2'])
    reject_stats.loc[reject_stats.part == p,
                     reject_stats.columns == 'cs-1'] = len(erp_cues['CS-1'])
    reject_stats.loc[reject_stats.part == p,
                     reject_stats.columns == 'cs-e'] = len(erp_cues['CS-E'])

    # Average across trials and plot
    figs_butter = []
    evokeds = dict()
    for cond in events_id.keys():
        evokeds[cond] = erp_cues[cond].average()
        figs_butter.append(pej(evokeds[cond],
                               title=cond,
                               show=False,
                               ts_args={'time_unit': 'ms'},
                               topomap_args={'time_unit': 'ms'}))

        evokeds[cond].save(opj(outdir, p + '_task-fearcond_' + cond
                               + '_ave.fif'))

    report.add_slider_to_section(figs_butter, captions=list(events_id.keys()),
                                 section='ERPs for cues',
                                 title='Butterfly plots for cues')

    # # Plot some channels and add to report
    chans_to_plot = ['Cz', 'Pz', 'CPz', 'Oz', 'Fz', 'FCz']
    figs_chan = []
    for c in chans_to_plot:
        pick = erp_cues.ch_names.index(c)
        figs_chan.append(mne.viz.plot_compare_evokeds(evokeds, picks=pick,
                                                      show=False))

    report.add_slider_to_section(figs_chan, captions=chans_to_plot,
                                 section='ERPs for cues', title='Cues/chans')

    # ________________________________________________________
    # Epoch around shock

    events_s = events[events.trigger_info == 'shock']
    events_s['cue_num'] = 255
    events_s['empty'] = 0
    events_shocks = np.asarray(events_s[['sample', 'empty', 'cue_num']])

    events_shocks_id = {
                        'shock': 255,
                        }

    erp_shocks = mne.Epochs(
                         raw,
                         events=events_shocks,
                         event_id=events_shocks_id,
                         tmin=param['erpbaseline'],
                         baseline=(param['erpbaseline'], 0),
                         tmax=param['erpepochend'],
                         preload=True,
                         verbose=False,
                         reject=param['erprejectshock'])

    nshocks = len(erp_shocks['shock'])
    reject_stats.loc[reject_stats.part == p,
                     reject_stats.columns
                     == 'perc_removed_shocks'] = ((54-nshocks)/54*100)
    reject_stats.loc[reject_stats.part == p,
                     reject_stats.columns == 'shocks'] = nshocks

    # Average across trials and plot
    figs_butter = []
    evokeds = dict()
    for cond in events_shocks_id.keys():
        evokeds[cond] = erp_shocks[cond].average()
        figs_butter.append(pej(evokeds[cond],
                               title=cond,
                               show=False,
                               ts_args={'time_unit': 'ms'},
                               topomap_args={'time_unit': 'ms'}))

        evokeds[cond].save(opj(outdir, p + '_task-fearcond_' + cond
                               + '_ave.fif'))

    report.add_figs_to_section(figs_butter,
                               captions=list(events_shocks_id.keys()),
                               section='ERPs for shocks')

    # Plot some channels and add to report
    figs_chan = []
    for c in chans_to_plot:
        pick = erp_shocks.ch_names.index(c)
        figs_chan.append(mne.viz.plot_compare_evokeds(evokeds, picks=pick,
                                                      show=False))

    report.add_slider_to_section(figs_chan, captions=chans_to_plot,
                                 section='ERPs for shocks',
                                 title='Shocks/chans')

    report.save(opj(outdir,  p + '_task-fearcond_erps_report.html'),
                open_browser=False, overwrite=True)

    # ________________________________________________________
    # Single trials for shocks
    events_c['trialsnum'] = range(1, 469)
    events_c['trials_name'] = ['trial_' + str(s).zfill(3)
                               for s in range(1, 469)]

    events_c['participant_id'] = p

    events_cues = np.asarray(events_c[['sample', 'empty', 'trialsnum']])

    trials_dict = dict()
    for idx, rows in events_c.iterrows():
        trials_dict[rows['trials_name']] = rows['trialsnum']

    erp_cues_single = mne.Epochs(
                         raw,
                         events=events_cues,
                         event_id=trials_dict,
                         tmin=param['erpbaseline'],
                         baseline=(param['erpbaseline'], 0),
                         tmax=param['erpepochend'],
                         metadata=events_c,
                         preload=True,
                         verbose=False)

    # Add bad trials to metadata
    strials_drop = erp_cues_single.copy()
    strials_drop.drop_bad(reject=param['erprejectshock'])
    badtrials = [1 if len(li) > 0 else 0 for li in strials_drop.drop_log]
    erp_cues_single.metadata['badtrial'] = badtrials

    # Save
    if erp_cues_single.info['sfreq'] != param['resamp']:
        erp_cues_single.resample(param['resamp'])
    erp_cues_single.save(opj(outdir, p
                             + '_task-fearcond_cues_singletrials-epo.fif'),
                         overwrite=True)

    # ________________________________________________________
    # Extract single trial values for shocks
    events_s['trialsnum'] = range(1, 55)
    events_s['trialsname'] = ['trial_' + str(t).zfill(3)
                              for t in range(1, 55)]

    events_s['participant_id'] = p
    events_cues = np.asarray(events_s[['sample', 'empty', 'trialsnum']])

    trials_dict = dict()
    for idx, rows in events_s.iterrows():
        trials_dict[rows['trialsname']] = rows['trialsnum']

    erp_shocks_single = mne.Epochs(
                                   raw,
                                   events=events_cues,
                                   event_id=trials_dict,
                                   tmin=param['erpbaseline'],
                                   baseline=(param['erpbaseline'], 0),
                                   tmax=param['erpepochend'],
                                   preload=True,
                                   metadata=events_s,
                                   verbose=False)

    # Add bad trials to metadata
    strials_drop = erp_shocks_single.copy()
    strials_drop.drop_bad(reject=param['erpreject'])
    badtrials = [1 if len(li) > 0 else 0 for li in strials_drop.drop_log]
    erp_shocks_single.metadata['badtrial'] = badtrials

    # Save
    if erp_shocks_single.info['sfreq'] != param['resamp']:
        erp_shocks_single.resample(param['resamp'])
    erp_shocks_single.save(opj(outdir, p
                               + '_task-fearcond_shock_singletrials-epo.fif'),
                           overwrite=True)

# Save rejection stats
reject_stats.to_csv(opj('/data/derivatives',
                        'task-fearcond_erps_rejectionstats.csv'))


reject_stats.describe().to_csv(opj('/data/derivatives',
                                   'task-fearcond_'
                                   + 'erps_rejectionstats_desc.csv'))


def average_time_win_strials(strials, chans_to_average, amp_lat):
    """Extract mean amplitude bewtween fixed latencies at specified channels

    Parameters
    ----------
    strials : mne Epochs
        MNE epochs data with metadata
    chans_to_average : list
        Channels to include in the average
    amp_lat : type
        Latencies of the segment to average

    Returns
    -------
    mne EPOCH
        epoch mne frame with metadata updated with column amplitude

    """

    for c in chans_to_average:
        for a in amp_lat:
            ampsepoch = strials.copy()
            # Crop epochs around latencies and drop unused channels
            ampsepoch.crop(tmin=a[0], tmax=a[1])
            ampsepoch.drop_channels([ch for ch in ampsepoch.ch_names
                                     if ch not in c])

            all_amps = []
            for idx, data in enumerate(ampsepoch):
                amp = np.average(data)
                all_amps.append(amp)

            # Normalize across trials
            all_amps = (all_amps - np.mean(all_amps))/np.std(all_amps)

            # Put in frame
            strials.metadata['amp_' + str(c) + '_' + str(a[0]) + '-'
                             + str(a[1])] = all_amps
    return strials


# Make single trials for cues
chans_to_average = [['Fz'], ['POz'], ['Cz'], ['CPz']]
amp_lat = [[0.4, 0.8]]   # Latencies boundaries to average between in s
all_meta = []
for p in part:
    outdir = opj('/data/derivatives',  p, 'eeg')
    epo = mne.read_epochs(opj(outdir, p
                              + '_task-fearcond_cues_singletrials-epo.fif'))

    epo = average_time_win_strials(epo, chans_to_average, amp_lat)

    blocks = [1]*36
    for i in range(2, 8):
        blocks = blocks + [i]*72

    epo.metadata['block'] = blocks
    epo.metadata['condblock_join'] = 0
    # Add condition-block joined
    for idx, row in enumerate(epo.metadata.iterrows()):
        row = row[1]
        if row.trial_cond4 == 'CS+' or row.trial_cond4 == 'CS+S':
            epo.metadata.loc[idx, ['condblock_join']] = ('CS+'
                                                         + str(int(row.block))
                                                         + '/CSE'
                                                         + str(int(row.block
                                                                   + 1)))

        if row.trial_cond4 == 'CS-E':
            epo.metadata.loc[idx, ['condblock_join']] = ('CS+'
                                                         + str(int(row.block
                                                                   - 1))
                                                         + '/CSE'
                                                         + str(int(row.block)))

        if row.trial_cond4 == 'CS-1':
            epo.metadata.loc[idx, ['condblock_join']] = ('CS-1'
                                                         + str(int(row.block))
                                                         + '/CS-2'
                                                         + str(int(row.block
                                                                   + 1)))
        if row.trial_cond4 == 'CS-2':
            epo.metadata.loc[idx, ['condblock_join']] = ('CS-1'
                                                         + str(int(row.block
                                                                   - 1))
                                                         + '/CS-2'
                                                         + str(int(row.block)))

    all_meta.append(epo.metadata)

all_meta = pd.concat(all_meta)
all_meta.to_csv('/data/derivatives/task-fearcond_erpsmeta.csv')
