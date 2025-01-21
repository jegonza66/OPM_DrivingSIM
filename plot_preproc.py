import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import os
import seaborn as sn
import pandas as pd

import functions_general
import paths
import save


def get_intervals_signals(reference_signal, signal_to_scale, fig=None):
    """
    Get the intervals of interest for scaling signals.
    Plot the two signals to scale in 2 subplots and interactively zoom in/out to the matching
    regions of interest of each signal to get the corresponding intervals.
    When ready, press Enter to continue.

    Parameters
    ----------
    reference_signal: ndarray
        The 1D reference signal with proper scaling.
    signal_to_scale: ndarray
        The 1D signal you wish to re-scale.
    fig: instance of figure, default None
        Figure to use for the plots. if None, figure is created.

    Returns
    -------
    axs0_start, axs0_end, axs1_start, axs1_end: int
      The axis start and end samples.
    """

    if fig == None:
        fig, axs = plt.subplots(2, 1)
    else:
        plt.close(fig)
        fig, axs = plt.subplots(2, 1)

    axs[0].plot(reference_signal)
    axs[0].set_title('EDF')
    axs[0].set_xlabel('Samples')
    axs[0].set_ylabel('Pixels')
    axs[1].plot(signal_to_scale)
    axs[1].set_title('MEG')
    axs[1].set_xlabel('Samples')
    axs[1].set_ylabel('Volts [\mu V]')
    fig.tight_layout()
    plt.pause(0.5)

    # Time to move around the plot to match the signals for scaling
    print('\nPlease arange the plots to matching parts of the signals. When ready, press Enter')
    while not plt.waitforbuttonpress():
        pass

    # Get plot limits for scaling signals in those ranges
    axs0_interval = [int(lim) for lim in axs[0].get_xlim()]
    axs1_interval = [int(lim) for lim in axs[1].get_xlim()]

    return fig, axs0_interval, axs1_interval


def scaled_signals(time, scaled_signals, reference_signals, interval_signal=None, interval_ref=None,
                   ref_offset=[0, 5500, 0], signal_offset=[0, 5500*1.2, 0], ylabels=['Gaze x', 'Gaze y', 'Pupil size'],
                   fig=None):
    """
    Plot scaled signals in selected interval into one plot for comparison and check scaling.

    Parameters
    ----------
    time: ndarray
        1D array of time for plot
    scaled_signal: list
        list of 1D scaled signals.
    reference_signal: list
        List of 1D reference signals with proper original scaling.
    interval_signal: {'list', 'tuple'}, default None
        The signal interval to use for scaling. if None, the whole signal is used. Default to None.
    interval_ref: {'list', 'tuple'}, default None
        The scaled signal interval to use for scaling. if None, the whole signal is used. Default to None.
    ref_offset: list, default [0, 5500, 0]
        List of offsets for the reference signal. Usually it is going to be 0 for Gaze x and Pupils size and an integer
        fot Gaze y, accounting for the offset between x and y Eyemaps.
    signal_offset: list, default [0, int(5500 * 1.2), 0]
        List of offsets for the scaled signal. Usually it is going to be 0 for Gaze x and Pupils size and an integer
        fot Gaze y, accounting for the offset between x and y Eyemaps.It differs from ref offset in the fact that
        signals might have different sampling rates.
    ylabels: list, default ['Gaze x', 'Gaze y', 'Pupil size']
        List of ylables to use in each subplot.
    fig: instance of figure, default None
        Figure to use for the plots. if None, figure is created.

    Returns
    ----------
    fig: instance of matplotlib figure
        The resulting figure
    """

    # Check inputs
    if len(scaled_signals) == len(reference_signals) == len(ref_offset) == len(signal_offset) == len(ylabels):
        num_subplots = len(scaled_signals)
    # If scaled and reference signals match in length, raise warning on the rest of the arguments
    elif len(scaled_signals) == len(reference_signals):
        num_subplots = len(scaled_signals)
        print(f'Lists: ref_offset, signal_offset, ylabels should have the same size, but have sizes:'
              f' {len(ref_offset)}, {len(signal_offset)}, {len(ylabels)}.\n'
              f'Using default values.')
        ref_offset = [0, 5500, 0][:num_subplots]
        signal_offset = [0, int(5500 * 1.2), 0][:num_subplots]
        ylabels = ['Gaze x', 'Gaze y', 'Pupil size'][:num_subplots]
    # If scaled and reference signals do not match in length, raise error
    else:
        raise ValueError(f'Lists: scaled_signal, reference_signal must have the same size, but have sizes: '
                         f'{len(scaled_signals)}, {len(reference_signals)}')

    # Make intervals to list because of indexing further ahead
    if not interval_signal:
        interval_signal = [None, None]
    if not interval_ref:
        interval_ref = [None, None]

    # If figure not provided, create instance of figure
    if not fig:
        fig, axs = plt.subplots(num_subplots, 1)
    else:
        plt.close(fig)
        fig, axs = plt.subplots(num_subplots, 1)

    # Set plot title
    plt.suptitle('Scaled and reference signals')
    # Iterate over signals ploting separately in subplots.
    for i, ax in enumerate(fig.axes):
        ax.plot(np.linspace(time[interval_ref[0]+ref_offset[i]]/1000, time[interval_ref[1]+ref_offset[i]]/1000,
                             interval_signal[1] - interval_signal[0]),
                 scaled_signals[i][interval_signal[0]+signal_offset[i]:interval_signal[1]+signal_offset[i]],
                 label='MEG')
        ax.plot(time[interval_ref[0]+ref_offset[i]:interval_ref[1]+ref_offset[i]]/1000,
                 reference_signals[i][interval_ref[0]+ref_offset[i]:interval_ref[1]+ref_offset[i]],
                 label='EDF')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(ylabels[i])
        if i==0:
            ax.legend(loc='upper right')

    fig.tight_layout()
    plt.pause(0.5)

    return fig


def alignment(subject, et_gazex, meg_gazex, corrs, et_block_start, meg_block_start, max_sample,
              et_block_end, meg_block_end, et_drop_start, meg_drop_start, block_num, block_trials, block_idxs,
              cross1_start_et, eyemap_start_et, eyemap_end_et):

    # Plot correlaion
    plt.ioff()
    plt.figure()
    plt.title(f'Block {block_num + 1}')
    plt.xlabel('Samples')
    plt.ylabel('Correlation')
    plt.plot(corrs)
    save_path = paths.plots_path + f'Preprocessing/{subject.subject_id}/ET_align/'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(save_path + f'Corr_block{block_num+1}.png')

    # Plot block gazex signals
    plot_samples_shift = max_sample + meg_drop_start - et_drop_start  # Samples shift for plot

    plt.figure(figsize=(15, 7))
    plt.plot(np.arange(len(et_gazex[et_block_start:et_block_end])) + plot_samples_shift, et_gazex[et_block_start:et_block_end], label='ET')
    plt.plot(np.arange(len(meg_gazex[meg_block_start:meg_block_end])), meg_gazex[meg_block_start:meg_block_end] * 200 + 1000, label='MEG')

    for i in range(block_trials):
        plt.vlines(x=cross1_start_et[block_idxs[i]] + plot_samples_shift - et_block_start, ymin=-500, ymax=1800,
                   color='black', linestyles='--', label='cross1')
    for i in range(5):
        plt.vlines(x=eyemap_start_et[i + 5*block_num] + plot_samples_shift - et_block_start, ymin=-500, ymax=1800,
                   color='green', linestyles='--', label='eyemap start')
        plt.vlines(x=eyemap_end_et[i + 5*block_num] + plot_samples_shift - et_block_start, ymin=-500, ymax=1800,
                   color='red', linestyles='--', label='eyemap end')

    plt.title(f'Block {block_num + 1}')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude [\mu V]')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.savefig(save_path + f'Signals_block{block_num + 1}.png')


def first_fixation_delay(subject, display_fig=False, save_fig=True):

    print('Plotting first fixation delay histogram')
    if display_fig:
        plt.ion()
    else:
        plt.ioff()

    fixations1_fix_screen = subject.fixations.loc[(subject.fixations['screen'].isin(['cross1', 'cross2'])) & (subject.fixations['n_fix'] == 1)]
    plt.figure()
    plt.hist(fixations1_fix_screen['delay'], bins=40)
    plt.title('1st fixation delay distribution')
    plt.xlabel('Time [s]')

    if save_fig:
        save_path = paths.plots_path + 'Preprocessing/' + subject.subject_id + '/'
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + f'{subject.subject_id} 1st fix delay dist.png')


def fixation_duration(subject, ax=None, display_fig=False, save_fig=True):

    print('Plotting fixation duration histogram')

    # Use provided axes (or not)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        display_fig = True
        save_fig = False

    if display_fig:
        plt.ion()
    else:
        plt.ioff()

    fixations_dur = subject.fixations['duration']

    ax.hist(fixations_dur, bins=100, range=(0, 1), edgecolor='black', linewidth=0.3, density=True, stacked=True)
    ax.set_title('Fixation duration')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Density')

    if save_fig:
        save_path = paths.plots_path + 'Preprocessing/' + subject.subject_id + '/'
        fname = f'{subject.subject_id} fix dur dist'
        save.fig(fig=fig, path=save_path, fname=fname)


def saccades_amplitude(subject, ax=None, display_fig=False, save_fig=True):

    print('Plotting saccades amplitude histogram')

    # Use provided axes (or not)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        display_fig = True
        save_fig = False

    if display_fig:
        plt.ion()
    else:
        plt.ioff()

    saccades_amp = subject.saccades['amp']

    ax.hist(saccades_amp, bins=100, range=(0, 20), edgecolor='black', linewidth=0.3, density=True, stacked=True)
    ax.set_title('Saccades amplitude')
    ax.set_xlabel('Amplitude (deg)')
    ax.set_ylabel('Density')

    if save_fig:
        save_path = paths.plots_path + 'Preprocessing/' + subject.subject_id + '/'
        fname = f'{subject.subject_id} sac amp'
        save.fig(fig=fig, path=save_path, fname=fname)


def saccades_dir_hist(subject, fig=None, axs=None, ax_idx=None, display_fig=False, save_fig=True):

    print('Plotting saccades direction histogram')

    # Use provided axes (or not)
    if axs is None or ax_idx is None:
        fig = plt.figure()
        ax = plt.subplot(polar=True)
    else:
        display_fig = True
        save_fig = False
        n_rows = axs.shape[0]
        n_cols = axs.shape[1]
        ax = axs.ravel()[ax_idx]
        ax.set_axis_off()
        ax = fig.add_subplot(n_rows, n_cols, ax_idx + 1, projection='polar')

    if display_fig:
        plt.ion()
    else:
        plt.ioff()

    saccades_deg = subject.saccades['deg']
    saccades_rad = saccades_deg * np.pi / 180

    n_bins = 24
    ang_hist, bin_edges = np.histogram(saccades_rad, bins=24, density=True)
    bin_centers = [np.mean((bin_edges[i], bin_edges[i+1])) for i in range(len(bin_edges) - 1)]

    bars = ax.bar(bin_centers, ang_hist, width=2*np.pi/n_bins, bottom=0.0, alpha=0.4, edgecolor='black')
    ax.set_xlabel('Saccades direction')
    ax.set_yticklabels([])

    for r, bar in zip(ang_hist, bars):
        bar.set_facecolor(plt.cm.Blues(r / np.max(ang_hist)))

    if save_fig:
        save_path = paths.plots_path + 'Preprocessing/' + subject.subject_id + '/'
        fname = f'{subject.subject_id} sac angular hist'
        save.fig(fig=fig, path=save_path, fname=fname)


def sac_main_seq(subject, hline=None, ax=None, display_fig=False, save_fig=True):

    print('Plotting main sequence')

    # Use provided axes (or not)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        display_fig = True
        save_fig = False

    if display_fig:
        plt.ion()
    else:
        plt.ioff()

    saccades_peack_vel = subject.saccades['peak_vel']
    saccades_amp = subject.saccades['amp']

    # ax.plot(saccades_amp, saccades_peack_vel, '.', alpha=0.1, markersize=2)

    # Logarithmic bins
    XL = np.log10(25)  # Adjusted to fit the xlim
    YL = np.log10(1000)  # Adjusted to fit the ylim
    # Create a 2D histogram with logarithmic bins
    ax.hist2d(saccades_amp, saccades_peack_vel, bins=[np.logspace(np.log10(0.2), XL, 300), np.logspace(np.log10(20), YL, 300)], cmap='Blues')
    ax.set_xlim(0.01)

    if hline:
        ax.hlines(y=hline, xmin=plt.gca().get_xlim()[0], xmax=plt.gca().get_xlim()[1], colors='grey', linestyles='--', label=hline)
        ax.legend()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('Main sequence')
    ax.set_xlabel('Amplitude (deg)')
    ax.set_ylabel('Peak velocity (deg)')
    ax.grid()

    # Set the limits of the axes
    ax.set_xlim(0.2, 25)
    ax.set_ylim(20, 1000)
    ax.set_aspect('equal')

    if save_fig:
        save_path = paths.plots_path + 'Preprocessing/' + subject.subject_id + '/'
        fname = f'{subject.subject_id} Sac Main sequence'
        save.fig(fig=fig, path=save_path, fname=fname)


def pupil_size_increase(subject, display_fig=False, save_fig=True):

    print('Plotting pupil size sincrease to different MSS')
    if display_fig:
        plt.ion()
    else:
        plt.ioff()

    fixations_pupil_s = subject.fixations.loc[(subject.fixations['screen'].isin(['cross1', 'ms', 'cross2'])) & (subject.fixations['n_fix'] == 1)]

    pupil_diffs = []
    mss = []
    for trial in subject.trial:
        trial_data = fixations_pupil_s.loc[fixations_pupil_s['trial'] == trial]

        try:
            if 'cross1' in trial_data['screen'].values:
                pupil_diff = trial_data[trial_data['screen'] == 'cross2']['pupil'].values[0] - trial_data[trial_data['screen'] == 'cross1']['pupil'].values[0]
            else:
                pupil_diff = trial_data[trial_data['screen'] == 'cross2']['pupil'].values[0] - \
                             trial_data[trial_data['screen'] == 'ms']['pupil'].values[0]
            pupil_diffs.append(pupil_diff)
            mss.append(trial_data['mss'].values[0])
        except:
            print(f'No cross1 or mss data in trial {trial}')

    plt.figure()
    sn.boxplot(x=mss, y=pupil_diffs)
    plt.title('Pupil size')
    plt.xlabel('MSS')
    plt.ylabel('Pupil size increase (fix point 2 - 1)')

    if save_fig:
        save_path = paths.plots_path + 'Preprocessing/' + subject.subject_id + '/'
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + f'{subject.subject_id} Pupil size increase.png')


def performance(subject, axs=None, display=False, save_fig=True):

    print('Plotting performance')

    # Use provided axes (or not)
    if axs is None:
        fig, axs = plt.subplots(2, sharex=True)
    else:
        display = True
        save_fig = False

    if display:
        plt.ion()
    else:
        plt.ioff()

    # Get response time mean and stf by MSS
    rt = subject.rt
    corr_ans = subject.corr_ans

    rt_1 = rt[np.where(subject.bh_data['Nstim'] == 1)[0]]
    rt_2 = rt[np.where(subject.bh_data['Nstim'] == 2)[0]]
    rt_4 = rt[np.where(subject.bh_data['Nstim'] == 4)[0]]

    rt1_mean = np.nanmean(rt_1)
    rt1_std = np.nanstd(rt_1)
    rt2_mean = np.nanmean(rt_2)
    rt2_std = np.nanstd(rt_2)
    rt4_mean = np.nanmean(rt_4)
    rt4_std = np.nanstd(rt_4)

    # Get correct ans mean and std by MSS
    corr_1 = corr_ans[np.where(subject.bh_data['Nstim'] == 1)[0]]
    corr_2 = corr_ans[np.where(subject.bh_data['Nstim'] == 2)[0]]
    corr_4 = corr_ans[np.where(subject.bh_data['Nstim'] == 4)[0]]

    corr1_mean = np.mean(corr_1)
    corr1_std = np.std(corr_1)
    corr2_mean = np.mean(corr_2)
    corr2_std = np.std(corr_2)
    corr4_mean = np.mean(corr_4)
    corr4_std = np.std(corr_4)

    # Plot
    axs[0].set_title(f'Performance {subject.subject_id}')

    axs[0].plot([1, 2, 4], [corr1_mean, corr2_mean, corr4_mean], 'o')
    axs[0].errorbar(x=[1, 2, 4], y=[corr1_mean, corr2_mean, corr4_mean], yerr=[corr1_std, corr2_std, corr4_std],
                    color='black', linewidth=0.5)
    axs[0].set_ylim([0, 1.3])
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xticks([1, 2, 4])

    axs[1].plot([1, 2, 4], [rt1_mean, rt2_mean, rt4_mean], 'o')
    axs[1].errorbar(x=[1, 2, 4], y=[rt1_mean, rt2_mean, rt4_mean], yerr=[rt1_std, rt2_std, rt4_std],
                    color='black', linewidth=0.5)
    axs[1].set_ylim([0, 10])
    axs[1].set_ylabel('Rt')
    axs[1].set_xlabel('MSS')
    axs[1].set_xticks([1, 2, 4])

    if save_fig:
        save_path = paths.plots_path + 'Preprocessing/' + subject.subject_id
        fname = f'/{subject.subject_id} Performance'
        save.fig(fig=fig, path=save_path, fname=fname)

    return corr1_mean, corr2_mean, corr4_mean, rt1_mean, rt2_mean, rt4_mean


def all_subj_performance(axs, all_acc, all_response_times, save_fig=False):
    # Use provided axes (or not)
    if axs is None:
        fig, axs = plt.subplots(2, sharex=True)
    else:
        display = True
        save_fig = False

    axs[0, 0].set_title(f'Performance')
    axs[0, 0].plot([1, 2, 4], [np.mean(all_acc[1]), np.mean(all_acc[2]), np.mean(all_acc[4])], 'o')
    axs[0, 0].errorbar(x=[1, 2, 4], y=[np.mean(all_acc[1]), np.mean(all_acc[2]), np.mean(all_acc[4])], yerr=[np.std(all_acc[1]), np.std(all_acc[2]), np.std(all_acc[4])],
                       color='black', linewidth=0.5)
    # axs[0, 0].set_ylim([0, 1])
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].set_xlabel('MSS')
    axs[0, 0].set_xticks([1, 2, 4])

    axs[1, 0].plot([1, 2, 4], [np.mean(all_response_times[1]), np.mean(all_response_times[2]), np.mean(all_response_times[4])], 'o')
    axs[1, 0].errorbar(x=[1, 2, 4], y=[np.mean(all_response_times[1]), np.mean(all_response_times[2]), np.mean(all_response_times[4])],
                       yerr=[np.std(all_response_times[1]), np.std(all_response_times[2]), np.std(all_response_times[4])],
                       color='black', linewidth=0.5)
    # axs[1, 0].set_ylim([0, 10])
    axs[1, 0].set_ylabel('Rt')
    axs[1, 0].set_xlabel('MSS')
    axs[1, 0].set_xticks([1, 2, 4])

    if save_fig:
        save_path = paths.plots_path + 'Preprocessing/All_Subjects'
        fname = f'/All_Subjects Performance'
        save.fig(fig=fig, path=save_path, fname=fname)

def trial_gaze(raw, subject, et_channels_meg, trial_idx, display_fig=False, save_fig=True):

    gaze_x = et_channels_meg[0]
    gaze_y = et_channels_meg[1]

    trial = trial_idx + 1

    plt.clf()
    plt.close('all')

    if display_fig:
        plt.ion()
    else:
        plt.ioff()

    # get trial info from bh data
    pres_abs_trial = 'Present' if subject.bh_data['Tpres'].astype(int)[trial_idx] == 1 else 'Absent'
    correct_trial = 'Correct' if subject.corr_ans.astype(int)[trial_idx] == 1 else 'Incorrect'
    mss = subject.bh_data['Nstim'][trial_idx]

    # Get trial start and end samples
    trial_start_idx = functions_general.find_nearest(raw.times, subject.cross1[trial_idx])[0] - 120 * 2
    trial_end_idx = functions_general.find_nearest(raw.times, subject.vsend[trial_idx])[0] + 120 * 6

    # Plot
    fig = plt.figure(figsize=(15, 5))
    plt.title(f'Trial {trial} - {pres_abs_trial} - {correct_trial} - MSS: {int(mss)}')

    # Gazes
    plt.plot(raw.times[trial_start_idx:trial_end_idx], gaze_x[trial_start_idx:trial_end_idx], label='X')
    plt.plot(raw.times[trial_start_idx:trial_end_idx], gaze_y[trial_start_idx:trial_end_idx] - 1000, 'black', label='Y')

    # Screens
    plt.axvspan(ymin=0, ymax=1, xmin=subject.cross1[trial_idx], xmax=subject.ms[trial_idx], color='grey',
                alpha=0.4, label='Fix')
    plt.axvspan(ymin=0, ymax=1, xmin=subject.ms[trial_idx], xmax=subject.cross2[trial_idx], color='red',
                alpha=0.4, label='MS')
    plt.axvspan(ymin=0, ymax=1, xmin=subject.cross2[trial_idx], xmax=subject.vs[trial_idx], color='grey',
                alpha=0.4, label='Fix')
    plt.axvspan(ymin=0, ymax=1, xmin=subject.vs[trial_idx], xmax=subject.vsend[trial_idx], color='green',
                alpha=0.4, label='VS')

    plt.xlabel('time [s]')
    plt.ylabel('Gaze')
    plt.grid()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    if save_fig:
        save_path = paths.plots_path + 'Preprocessing/' + subject.subject_id + f'/Gaze_Trials/'
        fname = f'Trial {trial}'
        save.fig(fig=fig, path=save_path, fname=fname)


def emap_gaze(raw, subject, et_channels_meg, block_num, display_fig=False, save_fig=True):

    gaze_x = et_channels_meg[0]
    gaze_y = et_channels_meg[1]

    emaps = subject.emap.iloc[block_num]

    plt.clf()
    plt.close('all')

    if display_fig:
        plt.ion()
    else:
        plt.ioff()

    for emap_trial in range(len(emaps)-1):

        emap_trial_name = emaps.keys()[emap_trial].split("_")[0]

        # Get fixations and saccades
        saccades_t = subject.saccades.loc[(subject.saccades['screen'] == f'emap_{emap_trial_name}') &
                                          (emaps[emap_trial] < subject.saccades['onset']) &
                                          (subject.saccades['onset'] < emaps[emap_trial+1])]
        fixations_t = subject.fixations.loc[(subject.fixations['screen'] == f'emap_{emap_trial_name}') &
                                            (emaps[emap_trial] < subject.fixations['onset']) &
                                            (subject.fixations['onset'] < emaps[emap_trial+1])]

        # Get trial start and end samples
        trial_start_idx = functions_general.find_nearest(raw.times, emaps[emap_trial])[0] - 120 * 2
        trial_end_idx = functions_general.find_nearest(raw.times, emaps[emap_trial + 1])[0] + 120 * 6

        # Plot
        fig = plt.figure(figsize=(15, 5))
        plt.title(f'Eyemap {emap_trial_name.upper()} {block_num}')

        # Gazes
        plt.plot(raw.times[trial_start_idx:trial_end_idx], gaze_x[trial_start_idx:trial_end_idx], label='X')
        plt.plot(raw.times[trial_start_idx:trial_end_idx], gaze_y[trial_start_idx:trial_end_idx] - 1000, 'black', label='Y')

        plot_max = np.nanmax(gaze_x[trial_start_idx:trial_end_idx])
        plot_min = np.nanmin(gaze_y[trial_start_idx:trial_end_idx] - 1000)

        # Screens
        for sac_idx, saccade in saccades_t.iterrows():
            plt.vlines(x=saccade['onset'], ymin=plot_min, ymax=plot_max, colors='red', linestyles='--', label='sac')

        for fix_idx, fixation in fixations_t.iterrows():
            plt.axvspan(ymin=0, ymax=1, xmin=fixation['onset'], xmax=fixation['onset']+fixation['duration'], color='green',
                        alpha=0.4, label='fix')

        plt.xlabel('time [s]')
        plt.ylabel('Gaze')
        plt.grid()
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')

        if save_fig:
            save_path = paths.plots_path + 'Preprocessing/' + subject.subject_id + f'/Gaze_Trials/'
            fname = f'Eyemap_{emap_trial_name}_{block_num}'
            save.fig(fig=fig, path=save_path, fname=fname)


def scanpath(raw, subject, et_channels_meg, items_pos, trial_idx,
             screen_res_x=1920, screen_res_y=1080, img_res_x=1280, img_res_y=1024, display_fig=False, save_fig=True):

    gaze_x = et_channels_meg[0]
    gaze_y = et_channels_meg[1]

    trial = trial_idx + 1

    fixations_vs = subject.fixations.loc[subject.fixations['screen'] == 'vs']
    saccades_vs = subject.saccades.loc[subject.saccades['screen'] == 'vs']

    plt.clf()
    plt.close('all')

    # Path to psychopy data
    exp_path = paths.experiment_path

    # Get trial
    fixations_t = fixations_vs.loc[fixations_vs['trial'] == trial]
    saccades_t = saccades_vs.loc[saccades_vs['trial'] == trial]
    item_pos_t = items_pos.loc[items_pos['folder'] == subject.trial_imgs[trial_idx]]

    # Get vs from trial
    vs_start_idx = functions_general.find_nearest(raw.times, subject.vs[trial_idx])[0]
    vs_end_idx = functions_general.find_nearest(raw.times, subject.vsend[trial_idx])[0]

    # Load search image
    img = mpimg.imread(exp_path + 'images/cmp_' + subject.trial_imgs[trial_idx] + '.jpg')

    # Load targets
    bh_data_trial = subject.bh_data.iloc[trial_idx]
    target_keys = ['st1', 'st2', 'st3', 'st4', 'st5']
    targets = bh_data_trial[target_keys]
    st1 = mpimg.imread(exp_path + targets[0])
    st2 = mpimg.imread(exp_path + targets[1])
    st3 = mpimg.imread(exp_path + targets[2])
    st4 = mpimg.imread(exp_path + targets[3])
    st5 = mpimg.imread(exp_path + targets[4])

    # Load correct vs incorrect
    correct_ans = subject.corr_ans[trial_idx]

    # Colormap: Get fixation durations for scatter circle size
    sizes = fixations_t['duration'] * 100
    # Define rainwbow cmap for fixations
    cmap = plt.cm.rainbow
    # define the bins and normalize
    if len(fixations_t):
        fix_num = fixations_t['n_fix'].values.astype(int)
        bounds = np.linspace(1, fix_num[-1] + 1, fix_num[-1] + 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Display image True or False
    if display_fig:
        plt.ion()
    else:
        plt.ioff()

    fig = plt.figure(figsize=(10, 9))
    plt.suptitle(f'Subject {subject.subject_id} - Trial {trial}')

    # Items axes
    ax1 = plt.subplot2grid((5, 5), (0, 0), colspan=1)
    ax2 = plt.subplot2grid((5, 5), (0, 1), colspan=1)
    ax3 = plt.subplot2grid((5, 5), (0, 2), colspan=1)
    ax4 = plt.subplot2grid((5, 5), (0, 3), colspan=1)
    ax5 = plt.subplot2grid((5, 5), (0, 4), colspan=1)

    # Image axis
    ax6 = plt.subplot2grid((5, 5), (1, 0), colspan=5, rowspan=3)

    # Remove ticks from items and image axes
    for ax in plt.gcf().get_axes():
        ax.set_xticks([])
        ax.set_yticks([])

    # Gaze axis
    ax7 = plt.subplot2grid((5, 5), (4, 0), colspan=5)

    # Targets
    for ax, st in zip([ax1, ax2, ax3, ax4, ax5], [st1, st2, st3, st4, st5]):
        ax.imshow(st)

    # Colour
    if correct_ans:
        for spine in ax5.spines.values():
            spine.set_color('green')
            spine.set_linewidth(3)
    else:
        for spine in ax5.spines.values():
            spine.set_color('red')
            spine.set_linewidth(3)

    # Fixations
    if len(fixations_t):
        ax6.scatter(fixations_t['mean_x'] - (screen_res_x - img_res_x) / 2,
                    fixations_t['mean_y'] - (screen_res_y - img_res_y) / 2,
                    c=fix_num, s=sizes, cmap=cmap, norm=norm, zorder=3)

    # Image
    ax6.imshow(img, zorder=0)

    # Target circles
    target = item_pos_t.loc[item_pos_t['istarget'] == 1]

    if len(target):
        if correct_ans:
            color = 'green'
        else:
            color = 'red'
        circle = plt.Circle((target['center_x'], target['center_y']), radius=70, color=color, fill=False)
        ax6.add_patch(circle)
        # ax6.scatter(target['center_x'], target['center_y'], s=1000, color=color, alpha=0.0, zorder=1)

    # Scanpath
    ax6.plot(gaze_x[vs_start_idx:vs_end_idx] - (1920 - 1280) / 2,
             gaze_y[vs_start_idx:vs_end_idx] - (1080 - 1024) / 2,
             '--', color='black', zorder=2)

    if len(fixations_t):
        PCM = ax6.get_children()[0]  # When the fixations dots for color mappable were ploted (first)
        cb = plt.colorbar(PCM, ax=ax6, ticks=[fix_num[0] + 1/2, fix_num[int(len(fix_num)/2)]+1/2, fix_num[-1]+1/2])
        cb.ax.set_yticklabels([fix_num[0], fix_num[int(len(fix_num)/2)], fix_num[-1]])
        cb.ax.tick_params(labelsize=10)
        cb.set_label('# of fixation', fontsize=13)

    # Gaze
    ax7.plot(raw.times[vs_start_idx:vs_end_idx], gaze_x[vs_start_idx:vs_end_idx], label='X')
    ax7.plot(raw.times[vs_start_idx:vs_end_idx], gaze_y[vs_start_idx:vs_end_idx], 'black', label='Y')

    plot_min, plot_max = ax7.get_ylim()

    for sac_idx, saccade in saccades_t.iterrows():
        ax7.vlines(x=saccade['onset'], ymin=plot_min, ymax=plot_max, colors='red', linestyles='--', label='sac')

    for fix_idx, fixation in fixations_t.iterrows():
        color = cmap(norm(fixation['n_fix']))
        ax7.axvspan(ymin=0, ymax=1, xmin=fixation['onset'], xmax=fixation['onset'] + fixation['duration'],
                    color=color, alpha=0.4, label='fix')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
    ax7.set_ylabel('Gaze')
    ax7.set_xlabel('Time [s]')

    if save_fig:
        save_path = paths.plots_path + 'Preprocessing/' + subject.subject_id + f'/Scanpaths/'
        fname = f'Trial{trial}'
        save.fig(fig=fig, path=save_path, fname=fname)


def ms_scanpath(raw, subject, et_channels_meg, trial_idx, ms_items_pos, display_fig=False, save_fig=True):

    # Clear all previous figures
    plt.clf()
    plt.close('all')

    trial = trial_idx + 1

    # Et tracker gaze data
    gaze_x = et_channels_meg[0]
    gaze_y = et_channels_meg[1]

    # Get fixations and saccades for MS screen
    fixations_ms = subject.fixations.loc[subject.fixations['screen'] == 'ms']
    saccades_ms = subject.saccades.loc[subject.saccades['screen'] == 'ms']

    # Path to psychopy data
    exp_path = paths.experiment_path

    # Get fixations and saccades for corresponding trial
    fixations_t = fixations_ms.loc[fixations_ms['trial'] == trial]
    saccades_t = saccades_ms.loc[saccades_ms['trial'] == trial]

    # Get items position information for trial
    trial_info = ms_items_pos.iloc[trial_idx]
    trial_items = {key: {'X': trial_info[f'X{idx + 1}'], 'Y': trial_info[f'Y{idx + 1}']} for idx, key in enumerate(trial_info.keys()) if
                   'st' in key and trial_info[key] != 'blank.png'}
    # Rename st5 as target
    if 'st5' in trial_items.keys():
        trial_items['target'] = trial_items.pop('st5')
    # Make dataframe to iterate over rows
    trial_items = pd.DataFrame(trial_items).transpose()

    # Get vs from trial
    ms_start_idx = functions_general.find_nearest(raw.times, subject.ms[trial_idx])[0]
    ms_end_idx = functions_general.find_nearest(raw.times, subject.cross2[trial_idx])[0]

    # Load targets
    bh_data_trial = subject.bh_data.iloc[trial_idx]
    items_keys = ['st1', 'st2', 'st3', 'st4', 'st5']
    item_images = bh_data_trial[items_keys]
    item_images['target'] = item_images.pop('st5')

    # Load correct vs incorrect
    correct_ans = subject.corr_ans[trial_idx]

    # Colormap: Get fixation durations for scatter circle size
    sizes = fixations_t['duration'] * 100
    # Define rainwbow cmap for fixations
    cmap = plt.cm.rainbow
    # define the bins and normalize
    if len(fixations_t):
        fix_num = fixations_t['n_fix'].values.astype(int)
        bounds = np.linspace(1, fix_num[-1] + 1, fix_num[-1] + 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Display image True or False
    if display_fig:
        plt.ion()
    else:
        plt.ioff()

    fig, axs = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]})
    plt.suptitle(f'Subject {subject.subject_id} - Trial {trial}')

    # Remove ticks from items and image axes
    for ax in plt.gcf().get_axes():
        ax.set_xticks([])
        ax.set_yticks([])

    # Fixations
    if len(fixations_t):
        axs[0].scatter(fixations_t['mean_x'],
                       fixations_t['mean_y'],
                       c=fix_num, s=sizes, cmap=cmap, norm=norm, zorder=3)

    gray_background_path = paths.experiment_path + 'gray1920x1080.png'
    gray_img = mpimg.imread(gray_background_path)
    axs[0].imshow(gray_img, zorder=0)

    from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
    for item_name, item in trial_items.iterrows():
        it_img = mpimg.imread(exp_path + item_images[item_name])

        imagebox = OffsetImage(it_img, zoom=0.35)
        ab = AnnotationBbox(imagebox, (item['X'],  item['Y']), frameon=False)
        axs[0].add_artist(ab)

        if item_name == 'target':
            if correct_ans:
                color = 'green'
            else:
                color = 'red'
            circle = plt.Circle((item['X'], item['Y']), radius=70, color=color, fill=False)
            axs[0].add_patch(circle)

    # Scanpath
    axs[0].plot(gaze_x[ms_start_idx:ms_end_idx],
             gaze_y[ms_start_idx:ms_end_idx],
             '--', color='black', zorder=2)

    if len(fixations_t):
        PCM = axs[0].get_children()[0]  # When the fixations dots for color mappable were ploted (first)
        cb = plt.colorbar(PCM, ax=axs[0], ticks=[fix_num[0] + 1/2, fix_num[int(len(fix_num)/2)]+1/2, fix_num[-1]+1/2])
        cb.ax.set_yticklabels([fix_num[0], fix_num[int(len(fix_num)/2)], fix_num[-1]])
        cb.ax.tick_params(labelsize=10)
        cb.set_label('# of fixation', fontsize=13)

    # Gaze
    axs[1].plot(raw.times[ms_start_idx:ms_end_idx], gaze_x[ms_start_idx:ms_end_idx], label='X')
    axs[1].plot(raw.times[ms_start_idx:ms_end_idx], gaze_y[ms_start_idx:ms_end_idx], 'black', label='Y')

    plot_min, plot_max = axs[1].get_ylim()

    for sac_idx, saccade in saccades_t.iterrows():
        axs[1].vlines(x=saccade['onset'], ymin=plot_min, ymax=plot_max, colors='red', linestyles='--', label='sac')

    for fix_idx, fixation in fixations_t.iterrows():
        color = cmap(norm(fixation['n_fix']))
        axs[1].axvspan(ymin=0, ymax=1, xmin=fixation['onset'], xmax=fixation['onset'] + fixation['duration'],
                    color=color, alpha=0.4, label='fix')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
    axs[1].set_ylabel('Gaze')
    axs[1].set_xlabel('Time [s]')

    if save_fig:
        save_path = paths.plots_path + 'Preprocessing/' + subject.subject_id + f'/MS_Scanpaths/'
        fname = f'Trial{trial}'
        save.fig(fig=fig, path=save_path, fname=fname)



## OLD out of use

def scanpath_BH(fixations, items_pos, bh_data, raw, gazex, gazey, subject, trial,
             screen_res_x=1920, screen_res_y=1080, img_res_x=1280, img_res_y=1024, display_fig=False, save=True):

    trial_idx = trial - 1
    fixations_vs = fixations.loc[fixations['screen'] == 'vs']

    plt.clf()
    plt.close('all')

    # Path to psychopy data
    exp_path = paths.experiment_path

    # Get trial
    fixations_t = fixations_vs.loc[fixations_vs['trial'] == trial]
    item_pos_t = items_pos.loc[items_pos['folder'] == subject.trial_imgs[trial_idx]]

    # Get vs from trial
    vs_start_idx = functions_general.find_nearest(raw.times, subject.vs[np.where(subject.trial == trial)[0]])[0]
    vs_end_idx = functions_general.find_nearest(raw.times, subject.onset[np.where(subject.trial == trial)[0]])[0]

    # Load search image
    img = mpimg.imread(exp_path + 'cmp_' + subject.trial_imgs[trial_idx] + '.jpg')

    # Load targets
    bh_data_trial = bh_data.loc[bh_data['searchimage'] == 'cmp_' + subject.trial_imgs[trial_idx] + '.jpg']
    target_keys = ['st1', 'st2', 'st3', 'st4', 'st5']
    targets = bh_data_trial[target_keys].values[0]
    st1 = mpimg.imread(exp_path + targets[0])
    st2 = mpimg.imread(exp_path + targets[1])
    st3 = mpimg.imread(exp_path + targets[2])
    st4 = mpimg.imread(exp_path + targets[3])
    st5 = mpimg.imread(exp_path + targets[4])

    # Load correct vs incorrect
    correct_ans = bh_data_trial['key_resp.corr'].values

    # Colormap: Get fixation durations for scatter circle size
    sizes = fixations_t['duration'] * 100
    # Define rainwbow cmap for fixations
    cmap = plt.cm.rainbow
    # define the bins and normalize
    fix_num = fixations_t['n_fix'].values.astype(int)
    bounds = np.linspace(1, fix_num[-1]+1, fix_num[-1]+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Display image True or False
    if display_fig:
        plt.ion()
    else:
        plt.ioff()

    plt.figure(figsize=(10, 9))
    plt.suptitle(f'Subject {subject.subject_id} - Trial {trial}')

    # Items axes
    ax1 = plt.subplot2grid((5, 5), (0, 0), colspan=1)
    ax2 = plt.subplot2grid((5, 5), (0, 1), colspan=1)
    ax3 = plt.subplot2grid((5, 5), (0, 2), colspan=1)
    ax4 = plt.subplot2grid((5, 5), (0, 3), colspan=1)
    ax5 = plt.subplot2grid((5, 5), (0, 4), colspan=1)

    # Image axis
    ax6 = plt.subplot2grid((5, 5), (1, 0), colspan=5, rowspan=3)

    # Remove ticks from items and image axes
    for ax in plt.gcf().get_axes():
        ax.set_xticks([])
        ax.set_yticks([])

    # Gaze axis
    ax7 = plt.subplot2grid((5, 5), (4, 0), colspan=5)

    # Targets
    for ax, st in zip([ax1, ax2, ax3, ax4, ax5], [st1, st2, st3, st4, st5]):
        ax.imshow(st)

    # Colour
    if correct_ans:
        for spine in ax5.spines.values():
            spine.set_color('green')
            spine.set_linewidth(3)
    else:
        for spine in ax5.spines.values():
            spine.set_color('red')
            spine.set_linewidth(3)

    # Fixations
    ax6.scatter(fixations_t['start_x'] - (screen_res_x - img_res_x) / 2,
                fixations_t['start_y'] - (screen_res_y - img_res_y) / 2,
                c=fix_num, s=sizes, cmap=cmap, norm=norm, zorder=3)

    # Image
    ax6.imshow(img, zorder=0)

    # Items circles
    ax6.scatter(item_pos_t['center_x'], item_pos_t['center_y'], s=1000, color='grey', alpha=0.5, zorder=1)
    target = item_pos_t.loc[item_pos_t['istarget'] == 1]

    # Target green/red
    if len(target):
        if correct_ans:
            ax6.scatter(target['center_x'], target['center_y'], s=1000, color='green', alpha=0.3, zorder=1)
        else:
            ax6.scatter(target['center_x'], target['center_y'], s=1000, color='red', alpha=0.3, zorder=1)

    # Scanpath
    ax6.plot(gazex[vs_start_idx:vs_end_idx] - (1920 - 1280) / 2,
             gazey[vs_start_idx:vs_end_idx] - (1080 - 1024) / 2,
             '--', color='black', zorder=2)


    PCM = ax6.get_children()[0]  # When the fixations dots for color mappable were ploted (first)
    cb = plt.colorbar(PCM, ax=ax6, ticks=[fix_num[0] + 1/2, fix_num[int(len(fix_num)/2)]+1/2, fix_num[-1]+1/2])
    cb.ax.set_yticklabels([fix_num[0], fix_num[int(len(fix_num)/2)], fix_num[-1]])
    cb.ax.tick_params(labelsize=10)
    cb.set_label('# of fixation', fontsize=13)

    # Gaze
    ax7.plot(raw.times[vs_start_idx:vs_end_idx], gazex[vs_start_idx:vs_end_idx], label='X')
    ax7.plot(raw.times[vs_start_idx:vs_end_idx], gazey[vs_start_idx:vs_end_idx], 'black', label='Y')
    ax7.legend(fontsize=8)
    ax7.set_ylabel('Gaze')
    ax7.set_xlabel('Time [s]')

    if save:
        save_path = paths.plots_path + 'Preprocessing/' + subject.subject_id + f'/Scanpaths/'
        os.makedirs(save_path + 'svg/', exist_ok=True)
        plt.savefig(save_path + f'Trial{trial}.png')
        plt.savefig(save_path + f'svg/Trial{trial}.svg')


def performance_BH(subject, display=False, save=True):

    if display:
        plt.ion()
    else:
        plt.ioff()

    # Get response time mean and stf by MSS
    bh_data = subject.bh_data
    bh_data = bh_data.loc[~pd.isna(bh_data['target.started'])].reset_index(drop=True)

    rt = bh_data['key_resp.rt']
    corr_ans = bh_data['key_resp.corr']

    rt_1 = rt[bh_data['Nstim'] == 1]
    rt_2 = rt[bh_data['Nstim'] == 2]
    rt_4 = rt[bh_data['Nstim'] == 4]

    rt1_mean = np.nanmean(rt_1)
    rt1_std = np.nanstd(rt_1)
    rt2_mean = np.nanmean(rt_2)
    rt2_std = np.nanstd(rt_2)
    rt4_mean = np.nanmean(rt_4)
    rt4_std = np.nanstd(rt_4)

    # Get correct ans mean and std by MSS
    corr_1 = corr_ans[bh_data['Nstim'] == 1]
    corr_2 = corr_ans[bh_data['Nstim'] == 2]
    corr_4 = corr_ans[bh_data['Nstim'] == 4]

    corr1_mean = np.mean(corr_1)
    corr1_std = np.std(corr_1)
    corr2_mean = np.mean(corr_2)
    corr2_std = np.std(corr_2)
    corr4_mean = np.mean(corr_4)
    corr4_std = np.std(corr_4)

    # Plot
    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle(f'Performance {subject.subject_id}')

    axs[0].plot([1, 2, 4], [corr1_mean, corr2_mean, corr4_mean], 'o')
    axs[0].errorbar(x=[1, 2, 4], y=[corr1_mean, corr2_mean, corr4_mean], yerr=[corr1_std, corr2_std, corr4_std],
                    color='black', linewidth=0.5)
    axs[0].set_ylim([0, 1.3])
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xticks([1, 2, 4])

    axs[1].plot([1, 2, 4], [rt1_mean, rt2_mean, rt4_mean], 'o')
    axs[1].errorbar(x=[1, 2, 4], y=[rt1_mean, rt2_mean, rt4_mean], yerr=[rt1_std, rt2_std, rt4_std],
                    color='black', linewidth=0.5)
    axs[1].set_ylim([0, 10])
    axs[1].set_ylabel('Rt')
    axs[1].set_xlabel('MSS')
    axs[1].set_xticks([1, 2, 4])

    if save:
        save_path = paths.plots_path + 'Preprocessing/' + subject.subject_id
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + f'/{subject.subject_id} Performance.png')


def line_noise_psd(subject, raw, filtered, display_fig=False, save_fig=True, fig_path=None, fig_name='RAW_PSD'):

    # Display image True or False
    if display_fig:
        plt.ion()
    else:
        plt.ioff()

    fig, axs = plt.subplots(nrows=2)
    plt.suptitle('Power line noise filtering')

    # Plot noise
    raw.plot_psd(picks='mag', ax=axs[0], show=display_fig)

    # Plot filtered
    filtered.plot_psd(picks='mag', ax=axs[1], show=display_fig)

    if save_fig:
        # Save
        if not fig_path:
            fig_path = paths.plots_path + 'Preprocessing/' + subject.subject_id + '/'
        save.fig(fig=fig, path=fig_path, fname=fig_name)