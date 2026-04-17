import pandas as pd
import paths
import setup
import save
import matplotlib.pyplot as plt


save_fig = True
exp_info = setup.exp_info()

all_fixations = pd.DataFrame()
all_saccades = pd.DataFrame()
all_bh_data = pd.DataFrame()
all_rt = pd.DataFrame()
all_mss = pd.DataFrame()
all_corr_ans = pd.DataFrame()

# Saccades, tgt fixations and response times relative to end time
all_saccades_end = pd.DataFrame()
all_fixations_target = pd.DataFrame()
all_response_times_end = pd.DataFrame()

all_acc = {1: [], 2: [], 4: []}
all_response_times = {1: [], 2: [], 4: []}

# Dictionary to store trials with target fixations on both screens
trials_with_both_tgt_fix = {}

for subject_id in exp_info.subjects_ids:

    # Load subject
    subject = setup.subject(subject_id=subject_id)
    print(f'\nSubject {subject.subject_id}')

    print(f'Total fixations: {len(subject.fixations())}')
    print(f'Total saccades: {len(subject.saccades())}')

    fixations = subject.fixations()
    saccades = subject.saccades()

    # Add subject identifier to fixations and saccades for proper grouping later
    fixations['subject'] = subject.subject_id
    saccades['subject'] = subject.subject_id

    all_fixations = pd.concat([all_fixations, fixations])
    all_saccades = pd.concat([all_saccades, saccades])


# Define all subjects class instance
subjects = setup.all_subjects(all_fixations, all_saccades, all_bh_data, all_rt.values, all_corr_ans.values, all_mss)


plt.rcParams.update({'font.size': 16})
fig, axs = plt.subplots(2, 2, figsize=(12, 7))

fixation_duration(subject=subjects, ax=axs[0, 0])
saccades_amplitude(subject=subjects, ax=axs[0, 1])
saccades_dir_hist(subject=subjects, fig=fig, axs=axs, ax_idx=2)
sac_main_seq(subject=subjects, ax=axs[1, 1])

fig.tight_layout()

if save_fig:
    save_path = paths.plots_path + 'Preprocessing/' + subjects.subject_id + '/'
    fname = f'{subjects.subject_id} Multipanel'
    save.fig(fig=fig, path=save_path, fname=fname)


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