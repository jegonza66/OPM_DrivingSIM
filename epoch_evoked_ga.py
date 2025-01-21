import os
import matplotlib.pyplot as plt
import mne
import functions_general
import functions_analysis
import load
import plot_general
import setup
import paths
from setup import subject

#----- Paths -----#
save_path = paths.save_path
plot_path = paths.plots_path
exp_info = setup.exp_info()


#----- Save data and display figures -----#
save_data = True
save_fig = True
display_figs = False
if display_figs:
    plt.ion()
else:
    plt.ioff()


#----- Parameters -----#
task = 'DA'
band_id = None  # Frequency band
epoch_id = 'DA'  # Epoch identifier
reject = False  # Peak to peak amplitude epoch rejection
data_type = 'ICA'  # 'RAW'

#----- Setup -----#

# Specific run path for saving data and plots
run_path = f'/Band_{band_id}/{epoch_id}_task{task}/'

# Save data paths
epochs_save_path = save_path + f'Epochs_{data_type}/' + run_path
evoked_save_path = save_path + f'Evoked_{data_type}/' + run_path
# Save figures paths
epochs_fig_path = plot_path + f'Epochs_{data_type}/' + run_path
evoked_fig_path = plot_path + f'Evoked_{data_type}/' + run_path

#----- Run -----#
evokeds = []
ga_subjects = []
for subject_id in exp_info.subjects_ids:

    # Load subject
    subject = setup.subject(subject_id=subject_id)

    tmin, tmax, plot_xlim = functions_general.get_time_lims(subject=subject, epoch_id=epoch_id)

    # Baseline duration
    baseline, plot_baseline = functions_general.get_baseline_duration(epoch_id=epoch_id, tmin=tmin, tmax=tmax)

    # Data filenames
    epochs_data_fname = f'Subject_{subject_id}_epo.fif'
    evoked_data_fname = f'Subject_{subject_id}_ave.fif'
    grand_avg_data_fname = f'Grand_average_ave.fif'

    try:
        # Load epoched data
        epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
    except:
        # Compute
        if band_id:
            meg_data = load.filtered_data(subject_id=subject_id, band_id=band_id, save_data=False)
        elif data_type == 'ICA':
            meg_data = load.ica_data(subject_id=subject_id, task=task)
        else:
            meg_data = load.preproc_meg_data(subject_id=subject_id)

        # Epoch data
        epochs, events = functions_analysis.epoch_data(subject=subject, epoch_id=epoch_id, meg_data=meg_data, tmin=tmin, tmax=tmax,
                                                       save_data=save_data, epochs_save_path=epochs_save_path,
                                                       epochs_data_fname=epochs_data_fname, reject=reject, baseline=baseline)

    #----- Evoked -----#
    # Define evoked and append for GA
    evoked = epochs.average()
    evokeds.append(evoked)
    ga_subjects.append(subject_id)

    # Separete MEG and misc channels
    evoked_meg = evoked.copy().pick('mag')

    # Plot evoked
    fname = 'Evoked_' + subject.subject_id
    plot_general.evoked(evoked_meg=evoked_meg, plot_xlim=plot_xlim, display_figs=display_figs, save_fig=save_fig,
                        fig_path=evoked_fig_path, fname=fname)

    if save_data:
        # Save evoked data
        os.makedirs(evoked_save_path, exist_ok=True)
        evoked.save(evoked_save_path + evoked_data_fname, overwrite=True)


# Compute grand average
grand_avg = mne.grand_average(evokeds)

# Save grand average
if save_data:
    os.makedirs(evoked_save_path, exist_ok=True)
    grand_avg.save(evoked_save_path + grand_avg_data_fname, overwrite=True)

# Separate MEG and misc channels
grand_avg_meg = grand_avg.copy().pick('mag')

# Plot evoked
fname = f'Grand_average'
# ylim = dict({'mag': (-150, 200)})
plot_general.evoked(evoked_meg=grand_avg_meg, plot_xlim=plot_xlim, display_figs=display_figs, save_fig=save_fig,
                    fig_path=evoked_fig_path, fname=fname)
