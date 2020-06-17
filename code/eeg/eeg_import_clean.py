# %% [markdown]

# TEST
# EEG preprocessing for Zoey's conditioning task
# @MP Coll, 2018, michelpcoll@gmail.com
# %%
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

# %%
###############################
# Parameters
##############################
layout = BIDSLayout('/data/source')
part = ['sub-' + s for s in layout.get_subject()]
part = ['sub-23']
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
         'filtertype': 'iir',
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
                         '38': [],  # None
                         '39': [],  # None
                         '40': [],  # None
                         '41': [],  # None
                         '42': ['CP2', 'P3'],
                         '43': [],  # None
                         '44': [],  # None
                         '45': [],  # None
                         '46': [],  # None
                         '47': [],  # None
                         '48': [],  # None
                         '49': ['T7'],
                         '50': [],  # None
                         '51': [],  # None
                         '52': [],  # None
                         '53': [],  # None
                         '54': [],  # None
                         '55': [],  # None
                         '56': ['Iz', 'P10'],
                         '57': ['Oz', 'O1']},
         # Visually identified bad ICA (For PCA == 40)
         'badica': {'23': [2],
                    '24': [0, 22],
                    '25': [1, 9],
                    '26': [0, 5, 15],
                    '27': [0],
                    '28': [0, 4],
                    '29': [0, 14],
                    '30': [0],
                    '31': [0, 4, 10, 15, 32],
                    '32': [2, 4, 7, 11, 17],
                    '33': [7, 16],
                    '34': [0, 2, 3, 6, 12, 15],
                    '35': [4, 5, 8],
                    '36': [0, 2, 30],
                    '37': [1, 6],
                    '38': [7, 14],
                    '39': [0, 8],
                    '40': [0, 3, 9, 10],
                    '41': [3, 4, 8, 25],
                    '42': [0, 15, 23, 26, 37],
                    '43': [3, 14, 16],
                    '44': [0, 18],
                    '45': [6, 20],
                    '46': [0, 4, 14, 16],
                    '47': [9],
                    '48': [4, 7, 8, 19],
                    '49': [20, 26],
                    '50': [0, 6],
                    '51': [0, 5],
                    '52': [4, 14],
                    '53': [5, 12],
                    '54': [3, 10, 14, 23],
                    '55': [3, 7, 23],
                    '56': [9, 11, 16, 19],
                    '57': [6, 8, 23]}
         }

# Output dir
outdir = opj('/data/derivatives')

# %%
# Choose participants to process

for p in part:

    ###############################
    # Initialise
    ##############################

    print('Processing participant'
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
    raw = mne.io.read_raw_bdf(f, montage=None, verbose=False,
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

    montage = mne.channels.read_montage(param['montage'],
                                        ch_names=raw.ch_names)

    raw.set_montage(montage)  # Apply positions
    raw.load_data()  # Load in RAM

    # _______________________________________________________________________
    # Bandpass filter
    raw.filter(
        param['hpfilter'],
        param['lpfilter'],
        method=param['filtertype'],
        filter_length='auto',
        phase='zero',
        fir_design='firwin',
        pad="reflect_limited")

    # ______________________________________________________________________
    # Plot filtered spectrum
    plt_psdf = raw.plot_psd(
        area_mode='range', tmax=10.0, average=False, show=False)
    report.add_figs_to_section(
        plt_psdf, captions='Filtered spectrum', section='Preprocessing')

    # ________________________________________________________________________
    # Remove and interpolate bad channels
    if param['visualinspect']:
        raw.plot(
            n_channels=raw.info['nchan'],
            scalings=dict(eeg=0.00020),
            block=True)

    raw.info['bads'] = param['badchannels'][p[-2:]]
    if raw.info['bads']:
        raw.interpolate_bads(reset_bads=True)

    # Plot sensor positions and add to report
    plt_sens = raw.plot_sensors(show_names=True, show=False)
    report.add_figs_to_section(
        plt_sens,
        captions='Sensor positions (bad in red)',
        section='Preprocessing')

    # ________________________________________________________________________
    # Clean with ICA

    # Make long epochs around trial onsets for ICA

    events['empty'] = 0
    events['triallabel'] = ['trial_' + str(i) for i in range(1, 469)]
    events['trialnum'] = range(1, 469)
    events_array = np.asarray(events[['sample', 'empty', 'trialnum']])

    alltrialsid = {}
    for idx, name in enumerate(list(events['triallabel'])):
        alltrialsid[name] = int(idx + 1)

    epochs_ICA = mne.Epochs(
        raw,
        events=events_array,
        event_id=alltrialsid,
        tmin=-2,
        baseline=None,
        tmax=2,
        preload=True,
        verbose=False)

    print('Processing ICA for part ' + p + '. This may take some time.')
    param
    ica = ICA(n_components=param['n_components'],
              method=param['icamethod'],
              random_state=param['random_state'])
    ica.fit(epochs_ICA)

    # Add topo figures to report
    plt_icacomp = ica.plot_components(show=False, res=32)
    for l in range(len(plt_icacomp)):
        report.add_figs_to_section(
            plt_icacomp[l], captions='ICA', section='Artifacts')

    # Identify which ICA correlate with eye blinks,
    eog_averagev = create_eog_epochs(raw, ch_name='Fp1',
                                     verbose=False).average()
    # Find EOG ICA via correlation
    eog_epochsv = create_eog_epochs(
        raw, ch_name='Fp1', verbose=False)  # get single EOG trials
    eog_indsv, scoresr = ica.find_bads_eog(
        eog_epochsv, ch_name='Fp1', verbose=False)  # find correlation

    # Get ICA identified in visual inspection
    icatoremove = param['badica'][p[-2:]]

    # Plot removed ICA and add to report
    figs = list()
    figs.append(ica.plot_sources(eog_averagev,
                                 exclude=icatoremove,
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
    for ical in range(len(ica._ica_names)):
        f = ica.plot_properties(epochs_ICA,
                                picks=ical,
                                psd_args={'fmax': 35.},
                                show=False)

        figs.append(f[0])
        capts.append(ica._ica_names[ical])

        figs.append(ica.plot_sources(eog_averagev,
                                     exclude=[ical],
                                     show=False))
        plt.close("all")

    report.add_slider_to_section(figs, captions=None,
                                 section='ICA-FULL')

    # Remove components manually identified
    ica.exclude.extend(icatoremove)

    # Apply ICA
    ica.apply(raw)

    # ______________________________________________________________________
    # Re-reference data
    raw, _ = mne.set_eeg_reference(raw, param['ref'], projection=False)

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
