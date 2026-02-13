import load
import matplotlib.pyplot as plt
import setup
import paths

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

meg_params = {'chs_id': 'mag',
              'band_id': None,
              'filter_sensors': True,
              'filter_method': 'iir',
              'data_type': 'ICA_annot',  # 'raw', 'ICA', 'processed', 'tsss'
              }


for subject_id in exp_info.subjects_ids:

    # Load subject
    subject = setup.subject(subject_id=subject_id)

    # Load MEG data
    meg_data = load.meg(subject_id=subject_id, meg_params=meg_params)

    # Load fixations
    fixations = subject.fixations()

    exp_info = setup.exp_info()

    # Fixations on left mirror
    fixations['left_mirror'] = ((fixations['mean_x'] > exp_info.left_mirror_px['x'][0]) &
                                (fixations['mean_x'] < exp_info.left_mirror_px['x'][1]) &
                                (fixations['mean_y'] > exp_info.left_mirror_px['y'][0]) &
                                (fixations['mean_y'] < exp_info.left_mirror_px['y'][1]))

    # Fixations on right mirror
    fixations['right_mirror'] = ((fixations['mean_x'] > exp_info.right_mirror_px['x'][0]) &
                                 (fixations['mean_x'] < exp_info.right_mirror_px['x'][1]) &
                                 (fixations['mean_y'] > exp_info.right_mirror_px['y'][0]) &
                                 (fixations['mean_y'] < exp_info.right_mirror_px['y'][1]))

    # Fixations on mirror
    fixations['on_mirror'] = fixations['left_mirror'] | fixations['right_mirror']

    # Fixations on stimulus
    stimulus_onsets = subject.master_df['symbol_onset_time'] #- meg_data.first_time
    stimulus_offsets = stimulus_onsets + exp_info.DA_duration #- meg_data.first_time

    fixations['onset_meg'] = fixations['onset'] + meg_data.first_time

    # Check if each fixation onset falls within any stimulus interval
    fixations['stimulus_present'] = False  # Initialize with 0 (no stimulus)
    fixations['stimulus_number'] = 0  # Initialize with 0 (no stimulus)
    for stim_number, (onset, offset) in enumerate(zip(stimulus_onsets, stimulus_offsets), start=1):
        # Mark fixations within this stimulus interval with the stimulus number
        mask = (fixations['onset_meg'] >= onset) & (fixations['onset_meg'] <= offset)
        fixations.loc[mask, 'stimulus_present'] = True
        fixations.loc[mask, 'stimulus_number'] = stim_number

    processed_save_path = paths.processed_path + subject_id + '/'
    # Save fixations ans saccades
    fixations.to_csv(processed_save_path + 'fixations.csv')

