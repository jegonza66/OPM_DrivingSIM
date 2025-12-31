import os
import functions_analysis
import functions_general
import mne
import mne.beamformer as beamformer
import save
import paths
import load
import setup
import numpy as np
import plot_general
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import itertools

# Load experiment info
exp_info = setup.exp_info()

#--------- Define Parameters ---------#

# Save
use_saved_data = False
save_fig = True
save_data = True
display_figs = False
if display_figs:
    plt.ion()
else:
    plt.ioff()

#----- Parameters -----#
task = 'DA'
# Trial selection
trial_params = {'epoch_id': 'fix',  # use'+' to mix conditions (red+blue)
                'reject': None,  # None to use default {'mag': 5e-12} / False for no rejection / 'subject' to use subjects predetermined rejection value
                'evt_from_df': True
                }

meg_params = {'chs_id': 'mag',
              'band_id': None,
              'filter_sensors': True,
              'filter_method': 'iir',
              'data_type': 'ICA_annot'
              }

# TRF parameters
trf_params = {'input_features': {#'fix': ['on_mirror', 'stimulus_present', 'on_mirror_X_stimulus_present'],  # _X_ for intersection between features
                                 'sac': None,
                                 #'pur': None,
                                 #'DAall': None,
                                 # 'left_but': None,
                                 # 'right_but': None
                                 },   # Select features (events)
              'standarize': False,
              'fit_power': False,
              'alpha': None,
              'tmin': -0.2,
              'tmax': 0.5,
              }
trf_params['baseline'] = (trf_params['tmin'], -0.05)

# Get frquencies
l_freq, h_freq = functions_general.get_freq_band(band_id=meg_params['band_id'])

# Compare features
run_comparison = True

# Source estimation parameters
force_fsaverage = False
model_name = 'lcmv'
surf_vol = 'surface'
spacing = 'ico4'
pos = 10  # Only for volume source estimation
pick_ori = None  # 'vector' For dipoles, 'max-power' for fixed dipoles in the direction tha maximizes output power
source_power = False
source_estimation = 'evk'  # 'epo' / 'evk' / 'cov' / 'trf'
visualize_alignment = False
active_times = None

# Baseline
if source_power or source_estimation == 'cov':
    bline_mode_subj = 'db'
else:
    bline_mode_subj = 'mean'
bline_mode_ga = 'mean'
plot_edge = 0.15

# Plot
initial_time = 0.09
difference_initial_time = 0.3
positive_cbar = None  # None for free determination, False to include negative values
plot_individuals = True
plot_ga = True

# Permutations test
run_permutations_GA = True
run_permutations_diff = False
desired_tval = 0.01
p_threshold = 0.05
mask_negatives = False


#--------- Setup ---------#

# Adapt to hfreq model
if meg_params['band_id'] == 'HGamma' or ((isinstance(meg_params['band_id'], list) or isinstance(meg_params['band_id'], tuple)) and meg_params['band_id'][0] > 40):
    model_name = 'hfreq-' + model_name


# --------- Freesurfer Path ---------#
# Define Subjects_dir as Freesurfer output folder
subjects_dir = os.path.join(paths.mri_path, 'freesurfer')
os.environ["SUBJECTS_DIR"] = subjects_dir

# Get Source space for default subject
if surf_vol == 'volume':
    fname_src = paths.sources_path + 'fsaverage' + f'/fsaverage_volume_{spacing}_{int(pos)}-src.fif'
elif surf_vol == 'surface':
    fname_src = paths.sources_path + 'fsaverage' + f'/fsaverage_surface_{spacing}-src.fif'
elif surf_vol == 'mixed':
    fname_src = paths.sources_path + 'fsaverage' + f'/fsaverage_mixed_{spacing}_{int(pos)}-src.fif'

src_default = mne.read_source_spaces(fname_src)

# Overwrite trial params to match trf features
if source_estimation == 'trf':

    # Define Grand average variables
    feature_evokeds = {}

    elements = trf_params['input_features'].keys()
    for feature in elements:
        feature_evokeds[feature] = []
        if isinstance(trf_params['input_features'], dict):
            try:
                for value in trf_params['input_features'][feature]:
                    feature_value = f'{feature}-{value}'
                    feature_evokeds[feature_value] = []
            except:
                pass

    trial_params['epoch_id'] = list(feature_evokeds.keys())

# Get param to compute difference from params dictionary
param_values = {key: value for key, value in trial_params.items() if type(value) == list}
# Exception in case no comparison
if param_values == {}:
    param_values = {list(trial_params.items())[0][0]: [list(trial_params.items())[0][1]]}

# Save source estimates time courses on FreeSurfer
stcs_default_dict = {}
GA_stcs = {}

# --------- Run ---------#
for param in param_values.keys():
    stcs_default_dict[param] = {}
    GA_stcs[param] = {}
    for param_value in param_values[param]:

        # Get run parameters from trial params including all comparison between different parameters
        run_params = trial_params
        # Set first value of parameters comparisons to avoid having lists in run params
        if len(param_values.keys()) > 1:
            for key in param_values.keys():
                run_params[key] = param_values[key][0]
        # Set comparison key value
        run_params[param] = param_value

        # Save source estimates time courses on default's subject source space
        stcs_default_dict[param][param_value] = []

        # Iterate over participants
        for sub_idx, subject_id in enumerate(exp_info.subjects_ids):
            # Load subject
            subject = setup.subject(subject_id=subject_id)

            # Get time windows from epoch_id name
            run_params['tmin'], run_params['tmax'], _ = functions_general.get_time_lims(subject=subject, epoch_id=run_params['epoch_id'])

            # Get baseline duration for epoch_id
            run_params['baseline'], run_params['plot_baseline'] = functions_general.get_baseline_duration(epoch_id=run_params['epoch_id'], tmin=run_params['tmin'],
                                                                                                          tmax=run_params['tmax'], plot_edge=plot_edge)

            # Run path
            run_path = f"Band_{meg_params['band_id']}/{run_params['epoch_id']}_task{task}_{run_params['tmin']}_{run_params['tmax']}_bline{run_params['baseline']}/"

            # Epochs path
            epochs_save_path = paths.save_path + f"Epochs_{meg_params['data_type']}/{run_path}/"
            cov_save_path = paths.save_path + f"Cov_Epochs_{meg_params['data_type']}/" + run_path

            # Source estimation path
            if surf_vol == 'volume':
                source_model_path = f"{model_name}_{surf_vol}_{spacing}_pos{pos}_{pick_ori}_{bline_mode_subj}_{source_estimation}_chs{meg_params['chs_id']}/"
            elif surf_vol == 'surface':
                source_model_path = f"{model_name}_{surf_vol}_{spacing}_{pick_ori}_{bline_mode_subj}_{source_estimation}_chs{meg_params['chs_id']}/"

            # Plots save paths
            fig_path = paths.plots_path + f"Source_Space_{meg_params['data_type']}/" + run_path + source_model_path

            # Source plots paths
            if source_power or source_estimation == 'cov':
                run_path = run_path.replace(f"{run_params['epoch_id']}_", f"{run_params['epoch_id']}_power_")

            # Rename fig path to specify plot times
            if source_estimation == 'cov':
                if active_times:
                    fig_path = fig_path.replace(f"{run_params['tmin']}_{run_params['tmax']}", f"{active_times[0]}_{active_times[1]}")
                else:
                    # Define active times
                    active_times = [0, run_params['tmax']]
                    fig_path = fig_path.replace(f"{run_params['tmin']}_{run_params['tmax']}", f"{active_times[0]}_{active_times[1]}")

            # --------- Coord systems alignment ---------#
            if force_fsaverage:
                subject_code = 'fsaverage'
                dig = False
            else:
                # Check if subject has MRI data
                try:
                    fs_subj_path = os.path.join(subjects_dir, subject.subject_id)
                    os.listdir(fs_subj_path)
                    dig = True
                    subject_code = subject_id
                except:
                    subject_code = 'fsaverage'
                    dig = False

            # Data filenames
            epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
            fname_lcmv = f'/{subject_code}_{meg_params['data_type']}_band{meg_params['band_id']}_{surf_vol}_{spacing}_pos{pos}_{pick_ori}-lcmv.h5'

            # Plot alignment visualization
            if visualize_alignment:
                plot_general.mri_meg_alignment(subject=subject, subject_code=subject_code, dig=dig, subjects_dir=subjects_dir)

            # Source data path
            sources_path_subject = paths.sources_path + subject.subject_id
            # Load forward model
            if surf_vol == 'volume':
                fname_fwd = sources_path_subject + f'/{subject_code}_{meg_params['data_type']}_chs{meg_params['chs_id']}_volume_{spacing}_{int(pos)}-fwd.fif'
            elif surf_vol == 'surface':
                fname_fwd = sources_path_subject + f'/{subject_code}_{meg_params['data_type']}_chs{meg_params['chs_id']}_surface_{spacing}-fwd.fif'
            elif surf_vol == 'mixed':
                fname_fwd = sources_path_subject + f'/{subject_code}_{meg_params['data_type']}_chs{meg_params['chs_id']}_mixed_{spacing}_{int(pos)}-fwd.fif'
            fwd = mne.read_forward_solution(fname_fwd)
            src = fwd['src']

            # Get epochs and evoked
            if os.path.exists(epochs_save_path + epochs_data_fname) and use_saved_data:
                # Load epochs
                epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
                # Pick channels
                picks = functions_general.pick_chs(chs_id=meg_params['chs_id'], info=epochs.info)
                epochs.pick(picks)

                if source_estimation == 'trf':
                    # Load MEG
                    meg_data = load.meg(subject_id=subject_id, meg_params=meg_params)

                    # Pick channels
                    picks = functions_general.pick_chs(chs_id=meg_params['chs_id'], info=meg_data.info)
                    meg_data = meg_data.pick(picks)

                    # Suppress warning about SSP projection
                    meg_data.info.normalize_proj()

                elif source_estimation == 'cov':
                    channel_types = epochs.get_channel_types()
                    bad_channels = epochs.info['bads']
                else:
                    # Define evoked from epochs
                    evoked = epochs.average()
                    evoked.pick(picks)

            else:
                # Load MEG
                meg_data = load.meg(subject_id=subject_id, meg_params=meg_params)

                # Pick channels
                picks = functions_general.pick_chs(chs_id=meg_params['chs_id'], info=meg_data.info)
                meg_data.pick(picks)

                # Suppress warning about SSP projection
                meg_data.info.normalize_proj()

                if source_estimation == 'cov':
                    channel_types = meg_data.get_channel_types()
                    bad_channels = meg_data.info['bads']

                else:
                    # Epoch data
                    epochs, events, onset_times = functions_analysis.epoch_data(subject=subject, epoch_id=run_params['epoch_id'], from_df=trial_params['evt_from_df'],
                                                                                     meg_data=meg_data, tmin=run_params['tmin'], tmax=run_params['tmax'],
                                                                                     baseline=run_params['baseline'], save_data=save_data,
                                                                                     epochs_save_path=epochs_save_path,
                                                                                     epochs_data_fname=epochs_data_fname, reject=run_params['reject'])

                    # Define evoked from epochs
                    evoked = epochs.average()
                    evoked.pick(picks)
                    epochs.pick(picks)

            # --------- Source estimation ---------#
            # Load filter
            if os.path.isfile(sources_path_subject + fname_lcmv) and use_saved_data:
                filters = mne.beamformer.read_beamformer(sources_path_subject + fname_lcmv)
            else:
                try:
                    meg_data
                except:
                    #  Make filters
                    meg_data = load.meg(subject_id=subject_id, meg_params=meg_params)
                    meg_data.pick(picks)
                    # Suppress warning about SSP projection
                    meg_data.info.normalize_proj()

                data_cov = mne.compute_raw_covariance(meg_data)
                filters = beamformer.make_lcmv(info=meg_data.info, forward=fwd, data_cov=data_cov, reg=0.05, pick_ori=pick_ori)
                filters.save(fname=sources_path_subject + fname_lcmv, overwrite=True)

            # Estimate sources from covariance matrix
            if source_estimation == 'cov':
                # Covariance method
                cov_method = 'shrunk'

                # Covariance matrix rank
                rank = sum([ch_type == 'mag' for ch_type in channel_types]) - len(bad_channels)
                if meg_params['data_type'] == 'ICA':
                    rank -= len(subject.ex_components)

                # Covariance fnames
                cov_baseline_fname = f"Subject_{subject.subject_id}_times{run_params['baseline']}_{cov_method}_{rank}-cov.fif"
                cov_act_fname = f'Subject_{subject.subject_id}_times{active_times}_{cov_method}_{rank}-cov.fif'

                stc = functions_analysis.estimate_sources_cov(subject=subject, meg_params=meg_params, trial_params=trial_params, filters=filters, active_times=active_times, rank=rank, bline_mode_subj=bline_mode_subj,
                                                              save_data=save_data, cov_save_path=cov_save_path, cov_act_fname=cov_act_fname,
                                                              cov_baseline_fname=cov_baseline_fname, epochs_save_path=epochs_save_path, epochs_data_fname=epochs_data_fname)

            # Estimate sources from epochs
            elif source_estimation == 'epo':
                # Define sources estimated on epochs
                stc_epochs = beamformer.apply_lcmv_epochs(epochs=epochs, filters=filters, return_generator=True)

                # Define stc object
                stc = beamformer.apply_lcmv(evoked=evoked, filters=filters)

                # Set data as zero to average epochs
                stc.data = np.zeros(shape=(stc.data.shape))
                for stc_epoch in stc_epochs:
                    data = stc_epoch.data
                    if source_power:
                        # Compute source power on epochs and average
                        if meg_params['band_id'] and not meg_params['filter_sensors']:
                            # Filter source data
                            data = functions_general.butter_bandpass_filter(data, band_id=meg_params['band_id'], sfreq=epochs.info['sfreq'], order=3)
                        # Compute envelope
                        analytic_signal = hilbert(data, axis=-1)
                        signal_envelope = np.abs(analytic_signal)
                        # Sum data of every epoch
                        stc.data += signal_envelope

                    else:
                        stc.data += data
                # Divide by epochs number
                stc.data /= len(epochs)

                if source_power:
                    # Drop edges due to artifacts from power computation
                    stc.crop(tmin=stc.tmin + plot_edge, tmax=stc.times.max() - plot_edge)

            # Estimate sources from evoked
            elif source_estimation == 'evk':
                # Apply filter and get source estimates
                stc = beamformer.apply_lcmv(evoked=evoked, filters=filters)

                if source_power:
                    # Compute envelope in source space
                    data = stc.data
                    if meg_params['band_id'] and not meg_params['filter_sensors']:
                        # Filter source data
                        data = functions_general.butter_bandpass_filter(data, band_id=meg_params['band_id'], sfreq=evoked.info['sfreq'], order=3)
                    # Compute envelope
                    analytic_signal = hilbert(data, axis=-1)
                    signal_envelope = np.abs(analytic_signal)
                    # Save envelope as data
                    stc.data = signal_envelope

                    # Drop edges due to artifacts from power computation
                    stc.crop(tmin=stc.tmin + plot_edge, tmax=stc.tmax - plot_edge)

            # Estimate sources from evoked
            elif source_estimation == 'trf':

                # # Pick channels
                # picks = functions_general.pick_chs(chs_id=meg_params['chs_id'], info=meg_data.info)
                # meg_data = meg_data.pick(picks)
                # print(len(meg_data.ch_names), 'channels selected for TRF source estimation')

                # Get trf paths
                trf_path = paths.save_path + (
                    f"TRF_{meg_params['data_type']}/Band_{meg_params['band_id']}/{trf_params['input_features']}_{trf_params['tmin']}_{trf_params['tmax']}_"
                    f"bline{trf_params['baseline']}_alpha{trf_params['alpha']}_std{trf_params['standarize']}/{meg_params['chs_id']}/").replace(":", "")
                trf_fig_path = trf_path.replace(paths.save_path, paths.plots_path)
                trf_fname = f'TRF_{subject.subject_id}.pkl'

                if os.path.exists(trf_path + trf_fname) and use_saved_data:
                    # Load TRF
                    rf = load.var(trf_path + trf_fname)
                    print('Loaded Receptive Field')

                else:
                    # Compute TRF for defined features
                    rf = functions_analysis.compute_trf(subject=subject, meg_data=meg_data, trial_params=trial_params, trf_params=trf_params, meg_params=meg_params,
                                                        features=list(feature_evokeds.keys()), alpha=trf_params['alpha'], use_saved_data=use_saved_data,
                                                        save_data=save_data,
                                                        trf_path=trf_path, trf_fname=trf_fname)

                # Get model coeficients as separate responses to each feature
                feature_evokeds = functions_analysis.parse_trf_to_evoked(subject=subject, rf=rf, meg_data=meg_data, feature_evokeds=feature_evokeds,
                                                                         trf_params=trf_params, meg_params=meg_params, sub_idx=sub_idx,
                                                                         plot_individuals=plot_individuals, save_fig=save_fig, fig_path=fig_path)

                # Get evoked from desired feature
                evoked = feature_evokeds[run_params['epoch_id']][sub_idx]

                # Apply filter and get source estimates
                stc = beamformer.apply_lcmv(evoked=evoked, filters=filters)

                if source_power:
                    # Compute envelope in source space
                    data = stc.data
                    if meg_params['band_id'] and not meg_params['filter_sensors']:
                        # Filter source data
                        data = functions_general.butter_bandpass_filter(data, band_id=meg_params['band_id'],
                                                                        sfreq=evoked.info['sfreq'], order=3)
                    # Compute envelope
                    analytic_signal = hilbert(data, axis=-1)
                    signal_envelope = np.abs(analytic_signal)
                    # Save envelope as data
                    stc.data = signal_envelope

                    # Drop edges due to artifacts from power computation
                    stc.crop(tmin=stc.tmin + plot_edge, tmax=stc.tmax - plot_edge)

            else:
                raise ValueError('No source estimation method was selected. Please select either estimating sources from evoked, epochs or covariance matrix.')

            if bline_mode_subj and not source_estimation == 'cov':
                # Apply baseline correction
                print(f"Applying baseline correction: {bline_mode_subj} from {run_params['baseline'][0]} to {run_params['baseline'][1]}")
                # stc.apply_baseline(baseline=baseline)  # mean
                if bline_mode_subj == 'db':
                    stc.data = 10 * np.log10(stc.data / np.expand_dims(stc.copy().crop(tmin=run_params['baseline'][0], tmax=run_params['baseline'][1]).data.mean(axis=-1), axis=-1))
                elif bline_mode_subj == 'ratio':
                    stc.data = stc.data / np.expand_dims(stc.copy().crop(tmin=run_params['baseline'][0], tmax=run_params['baseline'][1]).data.mean(axis=-1), axis=-1)
                elif bline_mode_subj == 'mean':
                    stc.data = stc.data - np.expand_dims(stc.copy().crop(tmin=run_params['baseline'][0], tmax=run_params['baseline'][1]).data.mean(axis=-1), axis=-1)

            if meg_params['band_id'] and source_power and not source_estimation == ' cov':
                # Filter higher frequencies than corresponding to nyquist of bandpass filter higher freq
                l_freq, h_freq = functions_general.get_freq_band(band_id=meg_params['band_id'])
                stc.data = functions_general.butter_lowpass_filter(data=stc.data, h_freq=h_freq/2, sfreq=evoked.info['sfreq'], order=3)

            # Morph to MNI152 space
            if subject_code != 'fsaverage':

                # Define morph function
                morph = mne.compute_source_morph(src=src, subject_from=subject_code, subject_to='fsaverage', src_to=src_default, subjects_dir=subjects_dir)

                # Apply morph
                stc_default = morph.apply(stc)

            else:
                stc_default = stc

            # Append to fs_stcs to make GA
            stcs_default_dict[param][param_value].append(stc_default)

            # Plot
            if plot_individuals:
                fname = f'{subject.subject_id}'
                plot_general.sources(stc=stc_default, src=src_default, subject='fsaverage', subjects_dir=subjects_dir, initial_time=initial_time, surf_vol=surf_vol,
                                     force_fsaverage=force_fsaverage, source_estimation=source_estimation, mask_negatives=mask_negatives,
                                     positive_cbar=positive_cbar, views=['lat', 'med'], save_fig=save_fig, save_vid=False, fig_path=fig_path, fname=fname)

        # Grand Average: Average evoked stcs from this epoch_id
        all_subj_source_data = np.zeros(tuple([len(stcs_default_dict[param][param_value])] + [size for size in stcs_default_dict[param][param_value][0].data.shape]))
        for j, stc in enumerate(stcs_default_dict[param][param_value]):
            all_subj_source_data[j] = stcs_default_dict[param][param_value][j].data
        if mask_negatives:
            all_subj_source_data[all_subj_source_data < 0] = 0

        # Define GA data
        GA_stc_data = all_subj_source_data.mean(0)

        # Copy Source Time Course from default subject morph to define GA STC
        GA_stc = stc_default.copy()

        # Reeplace data
        GA_stc.data = GA_stc_data
        GA_stc.subject = 'fsaverage'

        # Apply baseline on GA data
        if bline_mode_ga and not source_estimation == 'cov':
            print(f"Applying baseline correction: {bline_mode_ga} from {run_params['baseline'][0]} to {run_params['baseline'][1]}")
            # GA_stc.apply_baseline(baseline=baseline)
            if bline_mode_ga == 'db':
                GA_stc.data = 10 * np.log10(GA_stc.data / GA_stc.copy().crop(tmin=run_params['baseline'][0], tmax=run_params['baseline'][1]).data.mean(axis=1)[:, None])
            elif bline_mode_ga == 'ratio':
                GA_stc.data = GA_stc.data / GA_stc.copy().crop(tmin=run_params['baseline'][0], tmax=run_params['baseline'][1]).data.mean(axis=1)[:, None]
            elif bline_mode_ga == 'mean':
                GA_stc.data = GA_stc.data - GA_stc.copy().crop(tmin=run_params['baseline'][0], tmax=run_params['baseline'][1]).data.mean(axis=1)[:, None]

        # Save GA from epoch id
        GA_stcs[param][param_value] = GA_stc

        # --------- Plot GA ---------#
        if plot_ga:
            fname = f'GA'
            brain = plot_general.sources(stc=GA_stc, src=src_default, subject='fsaverage', subjects_dir=subjects_dir, initial_time=initial_time, surf_vol=surf_vol,
                                         force_fsaverage=force_fsaverage, source_estimation=source_estimation, mask_negatives=mask_negatives,
                                         positive_cbar=positive_cbar, views=['lat', 'med'], save_fig=save_fig, save_vid=False, fig_path=fig_path, fname=fname)

        # --------- Test significance compared to baseline --------- #
        if run_permutations_GA and pick_ori != 'vector':
            stc_all_cluster_vis, significance_voxels, significance_mask, t_thresh_name, time_label, p_threshold = \
                functions_analysis.run_source_permutations_test(src=src_default, stc=GA_stc, source_data=all_subj_source_data, subject='fsaverage', exp_info=exp_info,
                                                                save_regions=True, fig_path=fig_path, surf_vol=surf_vol, desired_tval=desired_tval, mask_negatives=mask_negatives,
                                                                p_threshold=p_threshold)

            # If covariance estimation, no time variable. Clusters are static
            if significance_mask is not None and source_estimation == 'cov':
                # Mask data
                GA_stc_sig = GA_stc.copy()
                GA_stc_sig.data[significance_mask] = 0

                # --------- Plot GA significant clusters ---------#
                fname = f'Clus_t{t_thresh_name}_p{p_threshold}'
                brain = plot_general.sources(stc=GA_stc_sig, src=src_default, subject='fsaverage', subjects_dir=subjects_dir, initial_time=0, surf_vol=surf_vol,
                                     time_label=time_label, force_fsaverage=force_fsaverage, source_estimation=source_estimation, views=['lat', 'med'],
                                     mask_negatives=mask_negatives, positive_cbar=positive_cbar, save_vid=False, save_fig=save_fig, fig_path=fig_path, fname=fname)

            # If time variable, visualize clusters using mne's function
            elif significance_mask is not None:
                fname = f'Clus_t{t_thresh_name}_p{p_threshold}'
                brain = plot_general.sources(stc=stc_all_cluster_vis, src=src_default, subject='fsaverage', subjects_dir=subjects_dir, initial_time=0,
                                             surf_vol=surf_vol, time_label=time_label, force_fsaverage=force_fsaverage, source_estimation=source_estimation,
                                             views=['lat', 'med'], mask_negatives=mask_negatives, positive_cbar=positive_cbar,
                                             save_vid=False, save_fig=save_fig, fig_path=fig_path, fname=fname)


#----- Difference between conditions -----#
for param in param_values.keys():
    if len(param_values[param]) > 1 and run_comparison:
        for comparison in list(itertools.combinations(param_values[param], 2)):

            if all(type(element) == int for element in comparison):
                comparison = sorted(comparison, reverse=True)

            # Figure difference save path
            if param == 'epoch_id':
                fig_path_diff = fig_path.replace(f'{param_values[param][-1]}', f'{comparison[0]}-{comparison[1]}')
            else:
                fig_path_diff = fig_path.replace(f'{param}{param_values[param][-1]}', f'{param}{comparison[0]}-{comparison[1]}')

            print(f'Taking difference between conditions: {param} {comparison[0]} - {comparison[1]}')

            # Get subjects difference
            stcs_diff = []
            for i in range(len(stcs_default_dict[param][comparison[0]])):
                stcs_diff.append(stcs_default_dict[param][comparison[0]][i] - stcs_default_dict[param][comparison[1]][i])

            # Average evoked stcs
            all_subj_diff_data = np.zeros(tuple([len(stcs_diff)]+[size for size in stcs_diff[0].data.shape]))
            for i, stc in enumerate(stcs_diff):
                all_subj_diff_data[i] = stcs_diff[i].data

            if mask_negatives:
                all_subj_diff_data[all_subj_diff_data < 0] = 0

            GA_stc_diff_data = all_subj_diff_data.mean(0)

            # Copy Source Time Course from default subject morph to define GA STC
            GA_stc_diff = GA_stc.copy()

            # Reeplace data
            GA_stc_diff.data = GA_stc_diff_data
            GA_stc_diff.subject = 'fsaverage'

            # --------- Plots ---------#
            if plot_ga:
                fname = f'GA'
                brain = plot_general.sources(stc=GA_stc_diff, src=src_default, subject='fsaverage', subjects_dir=subjects_dir, initial_time=difference_initial_time, surf_vol=surf_vol,
                                             force_fsaverage=force_fsaverage, source_estimation=source_estimation, mask_negatives=mask_negatives,
                                             views=['lat', 'med'], save_vid=False, save_fig=save_fig, fig_path=fig_path_diff, fname=fname, positive_cbar=positive_cbar)

            #--------- Cluster permutations test ---------#
            if run_permutations_diff and pick_ori != 'vector':
                stc_all_cluster_vis, significance_voxels, significance_mask, t_thresh_name, time_label, p_threshold = \
                    functions_analysis.run_source_permutations_test(src=src_default, stc=GA_stc_diff, source_data=all_subj_diff_data, subject='fsaverage',
                                                                    exp_info=exp_info, save_regions=True, fig_path=fig_path_diff, surf_vol=surf_vol, desired_tval=desired_tval,
                                                                    mask_negatives=mask_negatives, p_threshold=p_threshold)

                if significance_mask is not None and source_estimation == 'cov':
                    # Mask data
                    GA_stc_diff_sig = GA_stc_diff.copy()
                    GA_stc_diff_sig.data[significance_mask] = 0

                    # --------- Plots ---------#
                    fname = f'Clus_t{t_thresh_name}_p{p_threshold}'
                    brain = plot_general.sources(stc=GA_stc_diff_sig, src=src_default, subject='fsaverage', subjects_dir=subjects_dir, initial_time=0,
                                                 surf_vol=surf_vol, time_label=time_label, force_fsaverage=force_fsaverage, source_estimation=source_estimation,
                                                 views=['lat', 'med'], mask_negatives=mask_negatives, positive_cbar=positive_cbar,
                                                 save_vid=False, save_fig=save_fig, fig_path=fig_path_diff, fname=fname)

                elif significance_mask is not None:
                    fname = f'Clus_t{t_thresh_name}_p{p_threshold}'
                    brain = plot_general.sources(stc=stc_all_cluster_vis, src=src_default, subject='fsaverage', subjects_dir=subjects_dir, initial_time=0,
                                                 surf_vol=surf_vol, time_label=time_label, force_fsaverage=force_fsaverage, source_estimation=source_estimation,
                                                 views=['lat', 'med'], mask_negatives=mask_negatives, positive_cbar=positive_cbar,
                                                 save_vid=False, save_fig=save_fig, fig_path=fig_path_diff, fname=fname)
