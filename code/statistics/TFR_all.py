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
from mne.time_frequency import tfr_morlet, tfr_stockwell
from bids import BIDSLayout
import mne
import pandas as pd
import numpy as np
import os
from os.path import join as opj
import matplotlib.pyplot as plt
from bids import BIDSLayout
from mne.time_frequency import read_tfrs
import ptitprince as pt
import seaborn as sns
from mne.viz import plot_topomap
import scipy
from mne.stats import ttest_1samp_no_p
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
         'tfrepochstart': -2,
         'tfrepochend': 3,
         # Threshold to reject trials
         'tfr_reject': dict(eeg=500e-6),
         # TFR parameters
         'ttfreqs': np.arange(4, 41, 1),
         'n_cycles': 6,

         'testresampfreq': 512,
         # Njobs to run the TFR
         'njobs': 20,
         # Alpha Threshold
         'alpha': 0.05,
         # Font sizez in plot
         'titlefontsize': 24,
         'labelfontsize': 24,
         'ticksfontsize': 22,
         'legendfontsize': 20,
         # Excluded parts
         'excluded': ['sub-24', 'sub-31', 'sub-35', 'sub-51'],
         # Color palette
         'palette': ['#4C72B0', '#0d264f', '#55a868', '#c44e52'],
         # range on colormaps
         'pwrv': [-0.2, 0.2],
         # Njobs for permutations
         'njobs': 20,
         # Alpha Threshold
         'alpha': 0.05,
         # Number of permutations
         'nperms': 5000,
         # Random state to get same permutations each time
         'random_state': 23,
         # excluded participants
         'excluded': ['sub-24', 'sub-31', 'sub-35', 'sub-51'],

          }

cycles = [np.arange(4, 41, 1)/2]
for idx, cyc in enumerate(cycles):
    param['n_cycles'] = cyc
    idx = idx + 5
    outbase = opj('/data/derivatives' + str(idx))

    ##############################################################################
    # EPOCH AND TF transform
    ##############################################################################

    for p in part:
        indir = opj('/data/derivatives', p, 'eeg')
        # ______________________________________________________
        # Make out dir
        if not os.path.exists(outbase):
            os.mkdir(outbase)
        outdir = opj('/data/derivatives' + str(idx), p)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = opj('/data/derivatives' + str(idx), p, 'eeg')
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
        raw = mne.io.read_raw_fif(opj(indir,
                                      p + '_task-fearcond_cleanedeeg_raw.fif'),
                                  preload=True)

        # Load trial info in scr data
        events = pd.read_csv(layout.get(subject=p[-2:], extension='tsv',
                                        suffix='events',
                                        return_type='filename')[0], sep='\t')

        # Update samples stamp of events because some files were resampled
        evsamples = mne.find_events(raw)[:, 0][mne.find_events(raw)[:, 2]
                                               < 1000]
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
        # # Remove shocks
        # events_c = events_c[events_c['trial_type'] != 'CS+S']

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
                             decim=int(1024/param["testresampfreq"]),
                             n_jobs=param['njobs'],
                             average=False)



        tfr_cues_strials = None

        # # Apply baseline
        # strials.apply_baseline(mode=param['tfr_baseline_mode'],
        #                        baseline=param['tfr_baseline_time'])
        #
        # # Remove unused parts
        # strials.crop(tmin=param['erpbaseline'],
        #              tmax=param['erpepochend'])

        # strials.save(opj(outdir,  p + '_task-fearcond_'
        #                  + 'epochs-tfr.h5'), overwrite=True)

        # Drop bad
        strials_good = strials[np.where(np.asarray(goodtrials) == 1)[0]]
        strials = None
        induced = dict()
        for cond in events_id.keys():

            induced = strials_good[strials_good.metadata.trial_cond4
                                   == cond].average()
            induced.save(opj(outdir,  p + '_task-fearcond_' + cond
                             + '_avg-tfr.h5'), overwrite=True)

            chans_to_plot = ['Cz', 'Pz', 'CPz', 'Oz', 'Fz', 'FCz']
            figs_tfr = []
            for c in chans_to_plot:
                pick = induced.ch_names.index(c)
                figs_tfr.append(induced.plot(picks=[pick],
                                             tmin=-0.2, tmax=1,
                                             show=False,
                                             ))

            report.add_slider_to_section(figs_tfr,
                                         captions=chans_to_plot,
                                         section='TFR',
                                         title='Zscored TFR')

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
        report.save(opj(outdir,  p + '_task-fearcond'
                        + '_avg-tfr.html'),
                    open_browser=False, overwrite=True)
        plt.close('all')

    # Outpath for analysis
    outpath = opj(outbase, 'statistics')
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    outpath = opj(outbase, 'statistics', 'tfr_modelfree_anova')
    if not os.path.exists(outpath):
        os.mkdir(outpath)


    part = [p for p in part if p not in param['excluded']]

    ###########################################################################
    # Load and stack data
    ###########################################################################

    conditions = ['CS-1', 'CS-2', 'CS+', 'CS-E', ]
    anova_data = []
    data = dict()
    gavg = dict()
    for cond in conditions:
        pdat = []
        data[cond] = []

        for p in part:
            data[cond].append(read_tfrs(opj(outbase, p,
                                            'eeg',
                                            p + '_task-fearcond_' + cond
                                            + '_avg-tfr.h5'))[0])

            data[cond][-1].apply_baseline(mode='logratio',
                                          baseline=(-0.2, 0))

            data[cond][-1].crop(tmin=-0.2, tmax=1)

            pdat.append(np.float32(data[cond][-1].data))

        anova_data.append(np.stack(pdat))
        gavg[cond] = mne.grand_average(data[cond])

    anova_data = np.stack(anova_data)

    # # Take difference of interest for each part
    diff_data = np.empty((1,) + anova_data.shape[1:])
    diff1 = np.empty((1,) + anova_data.shape[1:])
    diff2 = np.empty((1,) + anova_data.shape[1:])

    for s in range(anova_data.shape[1]):
        diff_data[0, s, :, :, :] = ((anova_data[0, s, :, :, :]
                                     - anova_data[1, s, :, :, :])
                                    - (anova_data[2, s, :, :, :]
                                       - anova_data[3, s, :, :, :]))

    diff_data = np.squeeze(diff_data)

    # #########################################################################
    # ANOVA
    ##########################################################################
    # Always output time x freq x chan

    #  TFCE
    #
    # chan_connect, _ = mne.channels.find_ch_connectivity(data['CS-1'][0].info,
    #                                                     'eeg')
    # # Create a 1 ajacent frequency connectivity
    # freq_connect = (np.eye(len(data['CS-1'][0].freqs))
    #                 + np.eye(len(data['CS-1'][0].freqs), k=1)
    #                 + np.eye(len(data['CS-1'][0].freqs), k=-1))
    #
    # # Combine matrices to get a freq x chan connectivity matrix
    # connect = scipy.sparse.csr_matrix(np.kron(freq_connect,
    #                                           chan_connect.toarray())
    #                                   + np.kron(freq_connect,
    #                                             chan_connect.toarray()))

    #
    # def stat_fun(*args):  # Custom ANOVA for permutation
    #     return mne.stats.f_mway_rm(np.swapaxes(args, 1, 0),  # Swap sub and cond
    #                                factor_levels=[4],
    #                                effects='A',
    #                                return_pvals=False,
    #                                correction=True)[0]
    #
    #
    # tfce_dict = dict(start=0, step=0.2)
    #
    # # Reshape in cond, sub, time*freq, chan
    # anova_data_test = anova_data.swapaxes(2, 4)
    # shapea = anova_data_test.shape
    # anova_data_test = np.reshape(anova_data_test,
    #                              (shapea[0], shapea[1], shapea[2],
    #                               shapea[3]*shapea[4]))
    #
    # F_obs, clusters, pval, h0 = \
    #     mne.stats.permutation_cluster_test(anova_data_test,
    #                                        stat_fun=stat_fun,
    #                                        threshold=tfce_dict,
    #                                        tail=1,  # One tail cause anova
    #                                        n_permutations=param['nperms'],
    #                                        seed=23,
    #                                        connectivity=connect,
    #                                        max_step=1,
    #                                        n_jobs=param['njobs'],
    #                                        out_type='indices')

    # ##########################################################################
    # FDR
    shapea = anova_data.shape
    anova_data_uni = np.reshape(anova_data, (shapea[0], shapea[1],
                                             shapea[2]*shapea[3]*shapea[4]))
    F_obs, pval = mne.stats.f_mway_rm(np.swapaxes(anova_data_uni, 1, 0),
                                      factor_levels=[4],
                                      effects='A',
                                      return_pvals=True,
                                      correction=True)
    _, pval = mne.stats.fdr_correction(pval, alpha=param['alpha'])

    pvals = np.reshape(pval, (shapea[2],
                              shapea[3],
                              shapea[4]))
    F_obsout = np.reshape(F_obs, (shapea[2],
                                  shapea[3],
                                  shapea[4]))

    np.save(opj(outpath, 'cues4_tfr_anova_pvals.npy'), pvals)
    np.save(opj(outpath, 'cues4_tfr_anova_Fvals.npy'), F_obsout)
    np.save(opj(outpath, 'resamp_times.npy'), data['CS+'][0].times)
    np.save(opj(outpath, 'resamp_freqs.npy'), data['CS+'][0].freqs)

    # Difference
    # ##############################################################
    # FDR
    shapet = diff_data.shape
    testdata = np.reshape(diff_data, (shapet[0],
                                      shapet[1]*shapet[2]*shapet[3]))

    #
    tval = ttest_1samp_no_p(testdata, sigma=1e-3)
    pval = scipy.stats.t.sf(np.abs(tval), shapet[0]-1)*2  # two-sided pvalue
    #
    _, pval = mne.stats.fdr_correction(pval, alpha=param['alpha'])

    # ##############################################################
    # TFCE
    # from mne.stats import spatio_temporal_cluster_1samp_test as perm1samp
    # from functools import partial
    #
    # # T-test with hat correction
    # stat_fun_hat = partial(ttest_1samp_no_p, sigma=1e-3)
    #
    # testdata = diff_data.swapaxes(1, 3)
    # shapet = testdata.shape
    # testdata = np.reshape(diff_data, (shapet[0],
    #                                   shapet[1], shapet[2]*shapet[3]))
    #
    # # data is (n_observations, n_times, n_vertices)
    # tval, _, pval, _ = perm1samp(testdata,
    #                              n_jobs=param['njobs'],
    #                              threshold=dict(start=0, step=0.2),
    #                              connectivity=connect,
    #                              max_step=1,
    #                              check_disjoint=True,
    #                              n_permutations=100,
    #                              buffer_size=None)

    # ##############################################################

    # Reshape in chan x freq x time
    pvals = np.reshape(pval, (shapet[1],
                              shapet[2],
                              shapet[3]))
    tvals = np.reshape(tval, (shapet[1],
                              shapet[2],
                              shapet[3]))

    dat = data[cond][-1].copy()
    dat.data = tvals
    dat.data = np.where(pvals < 0.05, 1, 0)
    dat.plot_topomap()
    dat.plot('POz')

    np.save(opj(outpath, 'cuesdiff_tfr_ttest_pvals.npy'), pvals)
    np.save(opj(outpath, 'cuesduff_tfr_ttest_tvals.npy'), tvals)
    np.save(opj(outpath, 'resamp_times.npy'), data['CS+'][0].times)
    np.save(opj(outpath, 'resamp_freqs.npy'), data['CS+'][0].freqs)

    # #########################################################################
    # ERP analyses for Zoey's conditioning task
    # @MP Coll, 2018, michelpcoll@gmail.com
    ###########################################################################

    import mne
    import pandas as pd
    import numpy as np
    import os
    from os.path import join as opj
    import matplotlib.pyplot as plt
    from bids import BIDSLayout
    from mne.time_frequency import read_tfrs
    import ptitprince as pt
    import seaborn as sns
    from mne.viz import plot_topomap

    ###############################
    # Parameters
    ##############################
    layout = BIDSLayout('/data/source')
    # part = ['sub-' + s for s in layout.get_subject()]

    # Remove stupid pandas warning
    pd.options.mode.chained_assignment = None  # default='warn'

    # Outpath for analysis
    outpath = opj(outbase, 'statistics/tfr_modelfree_anova')
    # Outpath for figures
    outfigpath = opj(outbase, 'figures/tfr_modelfree_anova')

    if not os.path.exists(outpath):
        os.makedirs(outpath)
    if not os.path.exists(outfigpath):
        os.makedirs(outfigpath)

    # exclude
    part = [p for p in part if p not in param['excluded']]

    # Despine
    plt.rc("axes.spines", top=False, right=False)
    plt.rcParams['font.family'] = 'Arial Narrow'


    ###############################
    # Load data
    ##############################
    conditions = ['CS-1', 'CS-2', 'CS+', 'CS-E', ]
    anova_data = []
    data = dict()
    gavg = dict()
    for cond in conditions:
        pdat = []
        data[cond] = []

        for p in part:
            data[cond].append(read_tfrs(opj(outbase, p,
                                            'eeg',
                                            p + '_task-fearcond_' + cond
                                            + '_avg-tfr.h5'))[0])
            data[cond][-1].apply_baseline(mode='logratio',
                                          baseline=(-0.2, 0))

            data[cond][-1].crop(tmin=0, tmax=1, fmin=4, fmax=40)

            pdat.append(np.float32(data[cond][-1].data))

        anova_data.append(np.stack(pdat))
        gavg[cond] = mne.grand_average(data[cond])


    anova_data = np.stack(anova_data)

    # # Take difference of interest for each part
    diff_data = np.empty((2,) + anova_data.shape[1:])
    for s in range(anova_data.shape[1]):
        diff_data[0, s, ::] = (anova_data[0, s, :] - anova_data[1, s, :])
        diff_data[1, s, ::] = (anova_data[2, s, :] - anova_data[3, s, :])

    diff_data = np.squeeze(diff_data)

    # Always use  chan x freq x time

    pvals = np.load(opj(outpath, 'cues4_tfr_anova_pvals.npy'))
    Fvals = np.load(opj(outpath, 'cues4_tfr_anova_Fvals.npy'))
    p_plot = data[cond][0].copy()
    p_plot.data = pvals
    p_plot.data = np.where(p_plot.data < param['alpha'], 1, 0)

    # ###########################################################################
    # Make plot
    ###############################################################################


    # Helper functions
    def boxplot_freqs(foi, chan, time, gavg, data_all, ax, pal):
        # Colour palette for plotting
        c = 'CS-1'
        fidx = np.arange(np.where(gavg[c].freqs == foi[0])[0],
                         np.where(gavg[c].freqs == foi[1])[0])

        times = gavg[c].times
        tidx = np.arange(np.argmin(np.abs(times - time[0])),
                         np.argmin(np.abs(times - time[1])))
        cidx = gavg[c].ch_names.index(chan)

        plt_dat = data_all[:, :, cidx, :, :]
        plt_dat = plt_dat[:, :, fidx, :]
        plt_dat = plt_dat[:, :, :, tidx]
        plt_dat = np.average(plt_dat, 3)
        plt_dat.shape
        plt_dat = np.average(plt_dat, 2)

        plt_dat = pd.DataFrame(data=np.swapaxes(plt_dat, 1, 0),
                               columns=['CS-1', 'CS-2', 'CS-E', 'CS+'])
        plt_dat = pd.melt(plt_dat, var_name='Condition', value_name='Power')

        pt.half_violinplot(x='Condition', y="Power", data=plt_dat, inner=None,
                           jitter=True, color=".7", lwidth=0, width=0.6,
                           offset=0.17, cut=1, ax=ax,
                           linewidth=1, alpha=0.6, palette=pal, zorder=19)
        sns.stripplot(x='Condition', y="Power", data=plt_dat,
                      jitter=0.08, ax=ax,
                      linewidth=1, alpha=0.6, palette=pal, zorder=1)
        sns.boxplot(x='Condition', y="Power", data=plt_dat,
                    palette=pal, whis=np.inf, linewidth=1, ax=ax,
                    width=0.1, boxprops={"zorder": 10, 'alpha': 0.5},
                    whiskerprops={'zorder': 10, 'alpha': 1},
                    medianprops={'zorder': 11, 'alpha': 0.5})

        return ax


    # First line are the TF plots at cz
    for chan in ['Pz', 'POz', 'Cz', 'CPz', 'Fz']:
        fig = plt.figure(figsize=(18, 9))

        axes = []
        for i in [0, 2]:
            for j in [0, 2]:
                axes.append(plt.subplot2grid((4, 7),
                                             (i, j),
                                             colspan=2,
                                             rowspan=2))
        # Statistics
        axes.append(plt.subplot2grid((4, 7),
                                     (0, 5),
                                     colspan=2,
                                     rowspan=2))

        for idx, c in enumerate(conditions):
            pltdat = gavg[c]
            pick = pltdat.ch_names.index(chan)
            ch_mask = np.asarray([1 if c == chan else 0 for c in pltdat.ch_names])

            pltdat.plot(picks=[pick],
                        tmin=-0.5, tmax=1,
                        show=False,
                        cmap='viridis',
                        vmin=-0.3,
                        vmax=0.3,
                        title='',
                        axes=axes[idx],
                        colorbar=False,
                        )

            if idx < 2:
                axes[idx].set_xlabel('',
                                     fontdict={'fontsize': param['labelfontsize']})
                axes[idx].set_xticks([])
            else:
                axes[idx].set_xlabel('Time (s)',
                                     fontdict={'fontsize': param['labelfontsize']})
                axes[idx].tick_params(axis="x",
                                      labelsize=param['ticksfontsize'])
            if idx == 0:
                axes[idx].set_ylabel('',
                                     fontdict={'fontsize': param['labelfontsize']})
            else:
                axes[idx].set_ylabel(None,
                                     fontdict={'fontsize': param['labelfontsize']})

            if idx in [1, 3]:
                axes[idx].set_yticks([])
            else:
                axes[idx].tick_params(axis="y", labelsize=param['ticksfontsize'])

        for idx, c in enumerate(conditions):
            axes[idx].set_title(c, fontdict={"fontsize": param['titlefontsize']})

        # Pvalue plot
        p_plot.plot(picks=[pick],
                    tmin=-0.2, tmax=1,
                    show=False,
                    cmap='Greys',
                    vmin=0.1,
                    vmax=1.1,
                    title='',
                    axes=axes[len(conditions)],
                    colorbar=False,
                    )
        plt.tight_layout()
        axes[-1].set_ylabel(None,
                            fontdict={'fontsize': param['labelfontsize']})
        axes[-1].set_xlabel('Time (s)',
                            fontdict={'fontsize': param['labelfontsize']})

        pos = axes[-1].get_position()
        pos.y0 = pos.y0 - 0.23      # for example 0.2, choose your value
        pos.y1 = pos.y1 - 0.23
        pos.x0 = pos.x0 - 0.06     # for example 0.2, choose your value
        pos.x1 = pos.x1 - 0.06
        axes[-1].set_position(pos)

        axes[-1].set_title("FDR corrected at " + chan,
                           fontdict={"fontsize": param['titlefontsize']})
        axes[-1].tick_params(axis="both", labelsize=param['ticksfontsize'])

        fig.text(-0.01, 0.44, 'Frequency (Hz)',
                 fontdict={'fontsize': param['labelfontsize'],
                           'fontweight': 'normal'}, rotation=90)

        cax = fig.add_axes([0.58, 0.40, 0.01, 0.30],
                           label="cbar1")
        cbar1 = fig.colorbar(axes[0].images[0], cax=cax,
                             orientation='vertical', aspect=10)
        cbar1.set_label('Power (log baseline ratio)', rotation=-90,
                        labelpad=16, fontdict={'fontsize': param['labelfontsize']})
        cbar1.ax.tick_params(labelsize=param['ticksfontsize'])

        plt.savefig(opj(outfigpath, 'TF_plots_' + chan + '.png'),
                    bbox_inches='tight', dpi=600)

        # TOPO and bar plots
        plt.close('all')
        fig2 = plt.figure(figsize=(12, 6))

        axes = []

        for i in [0, 1]:
            for j in [0, 1]:
                axes.append(plt.subplot2grid((2, 4),
                                             (i, j),
                                             colspan=1,
                                             rowspan=1))

        for j in [0]:
            axes.append(plt.subplot2grid((2, 4),
                                         (j, 2),
                                         colspan=2,
                                         rowspan=2))

        foi = [20, 22]
        time = [0.6, 1]
        data_all = np.stack(anova_data)
        boxplot_freqs(foi, chan, time, gavg, data_all, axes[-1], param['palette'])
        axes[-1].set_xlabel('Condition',
                            fontdict={'fontsize': param['labelfontsize']})
        axes[-1].set_ylabel('Power (log baseline ratio)',
                            fontdict={'fontsize': param['labelfontsize']})
        axes[-1].tick_params(axis="both", labelsize=param['ticksfontsize'])
        # foi = [4, 6]
        # chan = 'Pz'
        # time = [0.2, 1]
        # boxplot_freqs(foi, chan, time, gavg, data_all, axes[-2], pal)

        foi = [20, 22]
        time = [0.6, 1]

        fidx = np.arange(np.where(gavg[c].freqs == foi[0])[0],
                         np.where(gavg[c].freqs == foi[1])[0])

        times = gavg[c].times
        tidx = np.arange(np.argmin(np.abs(times - time[0])),
                         np.argmin(np.abs(times - time[1])))

        p_dat = p_plot.data
        p_dat = p_dat[:, fidx, :]
        p_dat = p_dat[:, :, tidx]
        p_dat = np.average(p_dat, 2)
        p_dat = np.average(p_dat, 1)
        mask = np.where(p_dat > 0, 1, 0)

        for idx, c in enumerate(conditions):

            dcond = data_all[idx, :]
            plt_dat = dcond[:, :, fidx, :]
            plt_dat = plt_dat[:, :, :, tidx]
            plt_dat = np.average(plt_dat, 3)
            plt_dat = np.average(plt_dat, 2)

            plot_topomap(np.average(plt_dat, 0),
                         pltdat.info,
                         show=False,
                         cmap='viridis',
                         vmin=param['pwrv'][0],
                         vmax=param['pwrv'][1],
                         mask=None,
                         axes=axes[idx],
                         contours=False)

            axes[idx].set_title(c, fontdict={'fontsize': param['titlefontsize']})
        # Get data of interest

        # Second line Significance at Fz, Cz, Pz
        # Same with topo between sig freqs
        # Bar plot 20-40 Hz; 500-600 ms
        # Topo plot 20-40 Hz 500-600 ms
        plt.tight_layout()

        cax = fig2.add_axes([0.18, 0.52, 0.1, 0.05], label="cbar1")
        cbar1 = fig2.colorbar(axes[0].images[0], cax=cax,
                              orientation='horizontal', aspect=20)
        cbar1.set_label('Power (log baseline ratio)', rotation=0,
                        labelpad=14,
                        fontdict={'fontsize': param['labelfontsize']-5})
        cbar1.ax.tick_params(labelsize=param['ticksfontsize'])

        plt.savefig(opj(outfigpath, 'TF_topobar_' + chan + '.png'),
                    bbox_inches='tight', dpi=600)


    pvals = np.load(opj(outpath, 'cuesdiff_tfr_ttest_pvals.npy'))
    # pvals = np.swapaxes(pvals, 0, 2)

    # Same thing but for difference
    for chan in ['Pz', 'POz', 'Cz', 'CPz', 'Fz']:

        conditions = ['CS-1 vs CS-2', 'CS+ vs CS-E']

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        p_plot = data[cond][0].copy()
        p_plot.data = pvals
        p_plot.data = np.where(p_plot.data < param['alpha'], 1, 0)

        pow_plot = []
        pow_plot.append(data[cond][0].copy())
        pow_plot[0].data = np.mean(diff_data[0, ::], axis=0)
        pow_plot.append(data[cond][0].copy())
        pow_plot[1].data = np.mean(diff_data[1, ::], axis=0)

        for idx, c in enumerate(conditions):
            pltdat = pow_plot[idx]
            pick = pltdat.ch_names.index(chan)
            ch_mask = np.asarray([1 if c == chan else 0 for c in pltdat.ch_names])

            pltdat.plot(picks=[pick],
                        tmin=-0.5, tmax=1,
                        show=False,
                        cmap='viridis',
                        vmin=param['pwrv'][0],
                        vmax=param['pwrv'][1],
                        title='',
                        axes=axes[idx],
                        colorbar=False,
                        )

            axes[idx].tick_params(axis="x", labelsize=param['ticksfontsize'])

            if idx == 0:
                axes[idx].set_ylabel('Frequency (Hz)',
                                     fontdict={'fontsize': param['labelfontsize']})
                axes[idx].set_xlabel(None,
                                     fontdict={'fontsize': param['labelfontsize']})
            else:
                axes[idx].set_ylabel(None,
                                     fontdict={'fontsize': param['labelfontsize']})
                axes[idx].set_xlabel('Time (s)',
                                     fontdict={'fontsize': param['labelfontsize']})

            if idx in [1]:
                axes[idx].set_yticks([])
            else:
                axes[idx].tick_params(axis="y", labelsize=param['ticksfontsize'])

        for idx, c in enumerate(conditions):
            axes[idx].set_title(c, fontdict={"fontsize": param['titlefontsize']})

        # Pvalue plot
        p_plot.plot(picks=[pick],
                    tmin=-0.2, tmax=1,
                    show=False,
                    cmap='Greys',
                    vmin=0.1,
                    vmax=1.1,
                    title='',
                    axes=axes[len(conditions)],
                    colorbar=False
                    )
        plt.tight_layout()
        axes[-1].set_ylabel(None,
                            fontdict={'fontsize': param['labelfontsize']})
        axes[-1].set_xlabel('Time (s)',
                            fontdict={'fontsize': param['labelfontsize']})

        pos = axes[-1].get_position()
        pos.y0 = pos.y0
        pos.y1 = pos.y1
        pos.x0 = pos.x0 + 0.1
        pos.x1 = pos.x1 + 0.1
        axes[-1].set_position(pos)

        axes[-1].set_title("FDR corrected at " + chan,
                           fontdict={"fontsize": param['titlefontsize']})
        axes[-1].tick_params(axis="both", labelsize=param['ticksfontsize'])

        cax = fig.add_axes([0.68, 0.40, 0.01, 0.30],
                           label="cbar1")
        cbar1 = fig.colorbar(axes[0].images[0], cax=cax,
                             orientation='vertical', aspect=10)
        cbar1.set_label('Power (log baseline ratio)', rotation=-90,
                        labelpad=18, fontdict={'fontsize': param['labelfontsize']})
        cbar1.ax.tick_params(labelsize=param['ticksfontsize'])

        plt.savefig(opj(outfigpath, 'TF_plots_diff_' + chan + '.png'),
                    bbox_inches='tight', dpi=600)

        fig2, axes = plt.subplots(1, 3, figsize=(12, 4))

        foi = [20, 26]
        time = [0.6, 1]

        fidx = np.arange(np.where(gavg['CS-1'].freqs == foi[0])[0],
                         np.where(gavg['CS-1'].freqs == foi[1])[0])

        times = gavg['CS-1'].times
        tidx = np.arange(np.argmin(np.abs(times - time[0])),
                         np.argmin(np.abs(times - time[1])))
        cidx = gavg['CS-1'].ch_names.index(chan)
        plt_dat = np.average(diff_data[:, :, :, :, tidx], 4)
        plt_dat = np.average(plt_dat[:, :, :, fidx], 3)
        plt_dat = plt_dat[:, :, cidx]

        plt_dat = pd.DataFrame(data=np.swapaxes(plt_dat, 1, 0),
                               columns=conditions)
        plt_dat = pd.melt(plt_dat, var_name='Condition', value_name='Power')

        pt.half_violinplot(x='Condition', y="Power", data=plt_dat, inner=None,
                           jitter=True, color=".7", lwidth=0, width=0.6,
                           offset=0.17, cut=1, ax=axes[-1],
                           linewidth=1, alpha=0.6,
                           palette=[param['palette'][0], param['palette'][3]],
                           zorder=19)
        sns.stripplot(x='Condition', y="Power", data=plt_dat,
                      jitter=0.08, ax=axes[-1],
                      linewidth=1, alpha=0.6,
                      palette=[param['palette'][0], param['palette'][3]], zorder=1)
        sns.boxplot(x='Condition', y="Power", data=plt_dat,
                    palette=[param['palette'][0], param['palette'][3]],
                    whis=np.inf, linewidth=1,
                    ax=axes[-1],
                    width=0.1, boxprops={"zorder": 10, 'alpha': 0.5},
                    whiskerprops={'zorder': 10, 'alpha': 1},
                    medianprops={'zorder': 11, 'alpha': 0.5})
        axes[-1].set_xlabel('',
                            fontdict={'fontsize': param['labelfontsize']})
        axes[-1].set_ylabel('Power (log baseline ratio)',
                            fontdict={'fontsize': param['labelfontsize']})
        axes[-1].tick_params(axis="both", labelsize=param['ticksfontsize'])

        for idx, c in enumerate(conditions):
            dcond = diff_data[idx, :]
            fidx = np.arange(np.where(gavg['CS-1'].freqs == foi[0])[0],
                             np.where(gavg['CS-1'].freqs == foi[1])[0])

            times = gavg['CS-1'].times
            tidx = np.arange(np.argmin(np.abs(times - time[0])),
                             np.argmin(np.abs(times - time[1])))
            plt_dat = np.average(dcond[:, :, :, tidx], 3)
            plt_dat = np.average(plt_dat[:, :, fidx], 2)

            p_dat = pvals
            p_dat = np.average(p_dat[:, :, tidx], 2)
            p_dat = np.average(p_dat[:, fidx], 1)
            mask = np.where(p_dat < param['alpha'], 1, 0)

            plot_topomap(np.average(plt_dat, 0),
                         pltdat.info,
                         show=False,
                         cmap='viridis',
                         vmin=param['pwrv'][0],
                         vmax=param['pwrv'][1],
                         mask=mask,
                         axes=axes[idx],
                         contours=False)

            axes[idx].set_title(c, fontdict={'fontsize': param['titlefontsize']})
        # Get data of interest

        # Second line Significance at Fz, Cz, Pz
        # Same with topo between sig freqs
        # Bar plot 20-40 Hz; 500-600 ms
        # Topo plot 20-40 Hz 500-600 ms
        plt.tight_layout()

        cax = fig2.add_axes([0.27, 0.40, 0.1, 0.05], label="cbar1")
        cbar1 = fig2.colorbar(axes[0].images[0], cax=cax,
                              orientation='horizontal', aspect=20)
        cbar1.set_label('Power (log baseline ratio)', rotation=0,
                        labelpad=10,
                        fontdict={'fontsize': param['labelfontsize']-5})
        cbar1.ax.tick_params(labelsize=param['ticksfontsize'])

        plt.savefig(opj(outfigpath, 'TF_topobar_' + chan + '.png'),
                    bbox_inches='tight', dpi=600)
