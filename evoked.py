import load
import os
import mne
import matplotlib as mpl
mpl.use('Qt5Agg')  # Use Qt5Agg backend for interactive plotting
import matplotlib.pyplot as plt
import setup
import paths
import plot_general
import functions_general
import functions_analysis

save_path = paths.save_path
plot_path = paths.plots_path
exp_info = setup.exp_info()

#----- Save data and display figures -----#
save_data = True
save_fig = True
display_figs = True
use_saved_data = False
if display_figs:
    plt.ion()
else:
    plt.ioff()

#-----  Select frequency band -----#
#----- Parameters -----#
task = 'DA'
# Trial selection
trial_params = {'epoch_id': 'fixation',  # use'+' to mix conditions (red+blue)
                'reject': False,  # None to use default {'mag': 5e-12} / False for no rejection / 'subject' to use subjects predetermined rejection value
                'evt_from_df': True # If True, use events from df, otherwise use events from annotations
                }
meg_params = {'chs_id': 'mag',
              'band_id': None,
              'filter_sensors': True,
              'filter_method': 'iir',
              'data_type': 'ICA_annot',  # 'raw', 'ICA', 'processed', 'tsss'
              }

l_freq, h_freq = functions_general.get_freq_band(band_id=meg_params['band_id'])

# Plot eye movements
plot_gaze = False
reject = None

evokeds = []

for subject_id in exp_info.subjects_ids:

    # Load subject
    subject = setup.subject(subject_id=subject_id)

    # Get time windows from epoch_id name
    map = dict(hl_start={'tmin': -3, 'tmax': 35, 'plot_xlim': (-2.5, 33)})
    tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=trial_params['epoch_id'], subject=subject, plot_edge=0, map=map)

    # Baseline
    baseline, plot_baseline = functions_general.get_baseline_duration(epoch_id=trial_params['epoch_id'], tmin=tmin, tmax=tmax)

    # Specific run path for saving data and plots
    run_path = f'/Band_{meg_params['band_id']}/{trial_params['epoch_id']}_task{task}_{tmin}_{tmax}_bline{baseline}/'

    # Save data paths
    epochs_save_path = save_path + f'Epochs_{meg_params['data_type']}/' + run_path
    evoked_save_path = save_path + f'Evoked_{meg_params['data_type']}/' + run_path
    grand_avg_data_fname = f'Grand_average_ave.fif'

    # Save figures paths
    epochs_fig_path = plot_path + f'Epochs_{meg_params['data_type']}/' + run_path
    evoked_fig_path = plot_path + f'Evoked_{meg_params['data_type']}/' + run_path + f'{meg_params['chs_id']}/'

    # Data filenames
    epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
    evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'

    if os.path.exists(evoked_save_path + evoked_data_fname) and use_saved_data:
        # Load evoked data
        evoked = mne.read_evokeds(evoked_save_path + evoked_data_fname, verbose=False)[0]

    else:
        if os.path.exists(epochs_save_path + epochs_data_fname) and use_saved_data:
            # Load epoched data
            epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)

            # Get evoked by averaging epochs
            evoked = epochs.average(picks='mag')

            # Save data
            if save_data:
                # Save evoked data
                os.makedirs(evoked_save_path, exist_ok=True)
                evoked.save(evoked_save_path + evoked_data_fname, overwrite=True)
        else:
            # Load MEG
            meg_data = load.meg(subject_id=subject_id, meg_params=meg_params)

            # Epoch data
            epochs, events, onset_times = functions_analysis.epoch_data(subject=subject, epoch_id=trial_params['epoch_id'], meg_data=meg_data,
                                                           tmin=tmin, tmax=tmax, from_df=trial_params['evt_from_df'], save_data=save_data,
                                                           epochs_save_path=epochs_save_path,
                                                           epochs_data_fname=epochs_data_fname, reject=reject,
                                                           baseline=baseline)

            # ----- Evoked -----#
            # Define evoked and append for GA
            evoked = epochs.average()

        if save_data:
            # Save evoked data
            os.makedirs(evoked_save_path, exist_ok=True)
            evoked.save(evoked_save_path + evoked_data_fname, overwrite=True)

    # Apend to evokeds list to pass to grand average
    evokeds.append(evoked)

    # Separete MEG and misc channels
    picks = functions_general.pick_chs(chs_id=meg_params['chs_id'], info=evoked.info)
    evoked_meg = evoked.copy().pick(picks)

    # Save plot
    fname = 'Evoked_' + subject.subject_id + f'_{meg_params['chs_id']}'
    plot_general.evoked(evoked_meg=evoked_meg, plot_xlim=plot_xlim, display_figs=display_figs,
                        save_fig=save_fig, fig_path=evoked_fig_path, fname=fname)


# Compute grand average
grand_avg = mne.grand_average(evokeds, interpolate_bads=False)

# Save grand average
if save_data:
    grand_avg.save(evoked_save_path + grand_avg_data_fname, overwrite=True)

# Separete MEG and misc channels
picks = functions_general.pick_chs(chs_id=meg_params['chs_id'], info=grand_avg.info)
grand_avg = grand_avg.copy().pick(picks)

# Plot evoked
fname = f'Grand_average'
plot_general.evoked(evoked_meg=grand_avg, plot_xlim=plot_xlim, display_figs=display_figs,
                    save_fig=save_fig, fig_path=evoked_fig_path, fname=fname)