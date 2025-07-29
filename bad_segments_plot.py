import setup
import load
import plot_general
import functions_analysis
import save
import matplotlib.pyplot as plt


# Load experiment info
exp_info = setup.exp_info()
task = 'DA'
data_type = 'RAW_annot'
sds = 2
for subj_num, subject_id in enumerate(exp_info.subjects_ids):

    # Load subject
    subject = setup.subject(subject_id=subject_id)

    # Load TSSS data
    meg_data = load.meg_type(subject_id=subject_id, task=task, data_type=data_type.split('_')[0], preload=True)

    # Annotate bad segments as per default parameters
    annot_data, bad_segments = functions_analysis.annotate_bad_intervals(meg_data, sds=sds, data_type=None, data_fname=None,
                                                                         save_data=False)

    meg_data.get_channel_types()
    meg_data.pick('mag')

    fig, axs = plt.subplots(2, figsize=(12, 6))
    meg_data.compute_psd(fmax=100, reject_by_annotation=False).plot(axes=axs[0])

    # Plot bad segments
    fig = plot_general.bad_segments(meg_data=meg_data, bad_segments=bad_segments, sds=sds, ax=axs[1], fig=fig)

    # Save figure
    save.fig(fig=fig, path=setup.paths.plots_path + f'Preprocessing/{data_type}/',
             fname=f'{task}_{subject_id}_bad_segments_{sds}SD', save_svg=False)