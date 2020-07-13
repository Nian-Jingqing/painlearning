

# TEST
# EEG preprocessing for Zoey's conditioning task
# @MP Coll, 2018, michelpcoll@gmail.com

from mne.report import Report
import pprint
import mne
import os
from mne.preprocessing import ICA, create_eog_epochs
from os.path import join as opj
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bids import BIDSLayout

###############################
# Parameters
##############################
layout = BIDSLayout('/data/source')
part = ['sub-' + s for s in layout.get_subject()]
param = {
         # EOG channels
         'eogchan': ['EXG3', 'EXG4', 'EXG5'],
         # Empty channels to drop
         'dropchan': ['EXG6', 'EXG7', 'EXG8'],
         # Channels to rename
         'renamechan': {'EXG1': 'M1', 'EXG2': 'M2', 'EXG3': 'HEOGL',
                        'EXG4': 'HEOGR', 'EXG5': 'VEOGL'},
         # Montage to use
         'montage': 'standard_1005',
         # High pass filter cutoff
         'hpfilter': 0.1,
         # Low pass filter cutoff
         'lpfilter': 100,
         # Filter to use
         'filtertype': 'fir',
         # Plot for visual inspection (in Ipython, change pyplot to QT5)
         'visualinspect': False,
         # Reference
         'ref': 'average',
         # ICA parameters
         # Decimation factor before running ICA
         'icadecim': 4,
         # Set to get same decomposition each run
         'random_state': 23,
         # How many components keep in PCA
         'n_components': 40,
         # Reject trials exceeding this amplitude before ICA
         'erpreject': dict(eeg=500e-6),
         # Algorithm
         'icamethod': 'fastica',
         # Visually identified bad channels
         'badchannels': {'23': ['T7'],
                         '24': ['F7', 'FC3', 'FC4'],
                         '25': [],  # None
                         '26': ['P9'],
                         '27': [],
                         '28': [],
                         '29': ['F5'],
                         '30': [],
                         '31': ['P9'],
                         '32': [],
                         '33': [],
                         '34': ['T7'],
                         '35': ['P9', 'Iz', 'C5', 'CP3', 'CP5', 'F8'],
                         '36': [],  # None
                         '37': [],  # None
                         '38': ['P10'],  # None
                         '39': [],  # None
                         '40': [],  # None
                         '41': ['TP7'],  # None
                         '42': ['CP2', 'P3'],
                         '43': ['FC2'],  # None
                         '44': ['FC2'],  # None
                         '45': ['T7'],  # None
                         '46': [],  # None
                         '47': ['T8'],  # None
                         '48': [],  # None
                         '49': ['P7', 'P10', 'P9', 'T7', 'CP5', 'TP7', 'TP8'],
                         '50': [],  # None
                         '51': [],  # None
                         '52': ['P10'],  # None
                         '53': [],  # None
                         '54': [],  # None
                         '55': [],  # None
                         '56': ['P2', 'P10', 'AF7', 'Iz', 'P10'],
                         '57': ['Oz', 'O1']},
         # Visually identified bad ICAS
         'badica': {'23': [0, 4],
                    '24': [0, 2, 26],  # Excluded
                    '25': [2],
                    '26': [14],
                    '27': [0, 1, 2, 3],
                    '28': [0, 1, 2],
                    '29': [0],
                    '30': [0, 1],
                    '31': [0, 1, 10],  # Excluded
                    '32': [0, 3, 36, 65],
                    '33': [0, 1, 2, 3, 4, 5],
                    '34': [0, 33, 44],
                    '35': [0, 1, 2],  # Excluded
                    '36': [0, 10],
                    '37': [0, 43],
                    '38': [2],
                    '39': [0],
                    '40': [0, 14, 15, 16, 18, 21],
                    '41': [3, 7, 12, 29],
                    '42': [0, 2, 50, 54, 55],
                    '43': [0],
                    '44': [1],
                    '45': [0, 4, 54],
                    '46': [1, 15],
                    '47': [0, 3, 22, 23],
                    '48': [2, 8, 15],
                    '49': [0, 1],
                    '50': [0, 9],
                    '51': [0, 3],  # Excluded
                    '52': [0],
                    '53': [0, 8],
                    '54': [0, 46, 60],
                    '55': [0, 2],
                    '56': [0, 2],
                    '57': [0, 8, 9]}
         }

# Output dir
outdir = opj('/data/derivatives')


# Choose participants to process
for p in part:

    ###############################
    # Initialise
    ##############################

    print('Processing participant '
          + p)

    # _______________________________________________________
    # Make fslevel part dir
    pdir = opj(outdir, p, 'eeg')
    if not os.path.exists(pdir):
        os.mkdir(pdir)

    # _______________________________________________________
    # Initialise MNE report
    report = Report(verbose=False, subject=p,
                    title='EEG report for part ' + p)

    # report.add_htmls_to_section(
    #     htmls=part.comments[p], captions='Comments', section='Comments')
    report.add_htmls_to_section(
        htmls=pprint.pformat(param),
        captions='Parameters',
        section='Parameters')

    # ______________________________________________________
    # Load EEG file
    f = layout.get(subject=p[-2:], extension='bdf', return_type='filename')[0]
    raw = mne.io.read_raw_bdf(f, verbose=False,
                              eog=param['eogchan'],
                              exclude=param['dropchan'],
                              preload=True)


    # Rename external channels
    raw.rename_channels(param['renamechan'])

    events = pd.read_csv(layout.get(subject=p[-2:], extension='tsv',
                                    suffix='events',
                                    return_type='filename')[0], sep='\t')
    # For part with sampling at 2048 Hz, downsample
    if raw.info['sfreq'] == 2048:
        print('Resampling data to 1024 Hz')
        raw.resample(1024)

    # ______________________________________________________
    # Get events

    # Keep only rows for cues
    events = events[events['trial_type'].notna()]

    # Get events count
    events_count = events.trial_type.value_counts()

    # ReCalculate duration between events to double check
    events['time_from_prev2'] = np.insert((np.diff(events['sample'].copy()
                                                   / raw.info['sfreq'])),
                                          0, 0)

    events.to_csv(opj(pdir, p + '_task-fearcond_events.csv'))
    pd.DataFrame(events_count).to_csv(opj(pdir,
                                          p
                                          + '_task-fearcond_eventscount.csv'))

    # ______________________________________________________
    # Load and apply montage

    raw = raw.set_montage(param['montage'])
    raw.load_data()  # Load in RAM

    # ________________________________________________________________________
    # Remove bad channels
    if param['visualinspect']:
        raw.plot(
            n_channels=raw.info['nchan'],
            scalings=dict(eeg=0.00020),
            block=True)

    raw.info['bads'] = param['badchannels'][p[-2:]]

    # Plot sensor positions and add to report
    plt_sens = raw.plot_sensors(show_names=True, show=False)
    report.add_figs_to_section(
        plt_sens,
        captions='Sensor positions (bad in red)',
        section='Preprocessing')

    # _______________________________________________________________________
    # Bandpass filter
    raw_ica = raw.copy()  # Create a copy  to use different filter for ICA
    raw.filter(
        param['hpfilter'],
        None,
        method=param['filtertype'],
        verbose=True)

    # ______________________________________________________________________
    # Plot filtered spectrum
    plt_psdf = raw.plot_psd(
        area_mode='range', tmax=10.0, average=False, show=False)
    report.add_figs_to_section(
        plt_psdf, captions='Filtered spectrum', section='Preprocessing')

    # ________________________________________________________________________
    # Clean with ICA

    # Make epochs around trial for ICA

    events['empty'] = 0
    events['triallabel'] = ['trial_' + str(i) for i in range(1, 469)]
    events['trialnum'] = range(1, 469)
    events_array = np.asarray(events[['sample', 'empty', 'trialnum']])

    alltrialsid = {}
    for idx, name in enumerate(list(events['triallabel'])):
        alltrialsid[name] = int(idx + 1)

    # Low pass more agressively for ICA
    raw_ica.filter(l_freq=1, h_freq=100)
    epochs_ICA = mne.Epochs(
        raw_ica,
        events=events_array,
        event_id=alltrialsid,
        tmin=-0.5,
        baseline=None,
        tmax=1,
        preload=True,
        reject=param['erpreject'],
        verbose=False)

    print('Processing ICA for part ' + p + '. This may take some time.')
    ica = ICA(n_components=None,
              method=param['icamethod'],
              random_state=param['random_state'])
    ica.fit(epochs_ICA)

    # Add topo figures to report
    plt_icacomp = ica.plot_components(show=False, res=25)
    for l in range(len(plt_icacomp)):
        report.add_figs_to_section(
            plt_icacomp[l], captions='ICA', section='Artifacts')

    # Get manually identified bad ICA
    icatoremove = param['badica'][p[-2:]]

    # Identify which ICA correlate with eye blinks
    chaneog = 'VEOGL'
    eog_averagev = create_eog_epochs(raw_ica, ch_name=chaneog,
                                     verbose=False).average()
    # Find EOG ICA via correlation
    eog_epochsv = create_eog_epochs(
        raw_ica, ch_name=chaneog, verbose=False)  # get single EOG trials
    eog_indsv, scoresr = ica.find_bads_eog(
        eog_epochsv, ch_name=chaneog, verbose=False)  # find correlation

    fig = ica.plot_scores(scoresr, exclude=eog_indsv, show=False)
    report.add_figs_to_section(fig, captions='Correlation with EOG',
                               section='Artifact')

    # Get ICA identified in visual inspection
    figs = list()
    # Plot removed ICA and add to report
    ica.exclude = icatoremove
    figs.append(ica.plot_sources(eog_averagev,
                                 show=False,
                                 title='ICA removed on eog epochs'))

    report.add_figs_to_section(figs, section='ICA',
                               captions='Removed components '
                               + 'highlighted')

    report.add_htmls_to_section(
        htmls="IDX of removed ICA: " + str(icatoremove),
        captions='ICA-Removed',
        section='Artifacts')

    report.add_htmls_to_section(htmls="Number of removed ICA: "
                                + str(len(icatoremove)), captions="""ICA-
                                Removed""", section='Artifacts')

    # Loop all ICA and make diagnostic plots for report
    figs = list()
    capts = list()

    f = ica.plot_properties(epochs_ICA,
                            picks='all',
                            psd_args={'fmax': 35.},
                            show=False)

    for ical in range(len(ica._ica_names)):

        figs.append(f[ical])
        capts.append(ica._ica_names[ical])

        ica.exclude = [ical]
        figs.append(ica.plot_sources(eog_averagev,
                                     show=False))
        plt.close("all")

    f = None
    report.add_slider_to_section(figs, captions=None,
                                 section='ICA-FULL')

    # Remove components manually identified
    ica.exclude = icatoremove

    # Apply ICA
    ica.apply(raw)

    # ______________________________________________________________________
    # Re-reference data
    raw, _ = mne.set_eeg_reference(raw, param['ref'], projection=False)

    if raw.info['bads']:
        raw.interpolate_bads(reset_bads=True)
    # ______________________________________________________________________
    # Save cleaned data
    raw.save(opj(pdir, p + '_task-fearcond_cleanedeeg_raw.fif'),
             overwrite=True)

    #  _____________________________________________________________________
    report.save(opj(pdir, p + '_task-fearcond_importclean_report.html'),
                open_browser=False, overwrite=True)


# For manuscript
stats_dict = {}

# Number of bad channels
nbc = []
for p, bc in param['badchannels'].items():
    nbc.append(len(bc))

nbic = []
for p, bic in param['badica'].items():
    nbic.append(len(bic))

stats_dict = pd.DataFrame()
stats_dict['sub'] = part
stats_dict['n_bad_ica'] = nbic
stats_dict['n_bad_channels'] = nbc

stats_dict.describe().to_csv(opj('/data/derivatives/',
                                 'task-fearcond_importclean_stats.csv'))
