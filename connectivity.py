import os
import functions_analysis
import functions_general
import mne
import mne.beamformer as beamformer
import paths
import load
import setup
import mne_connectivity
import plot_general
import matplotlib.pyplot as plt
import numpy as np
import save
import mne_rsa
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import fdrcorrection
import itertools


# Removing NANs from meg_data
# meg_data.crop(tmin=0, tmax=meg_data.times[np.where(~np.isnan(meg_data._data[0]))[0][-1]])
# meg_data.save(paths.ica_path + f'{task}_{subject_id}_raw_ica_hfc_meg.fif', overwrite=True)

# Load experiment info
exp_info = setup.exp_info()

# --------- Define Parameters ---------#
save_fig = True
display_figs = False
plot_individuals = True
save_data = True

# Trial parameters
task = 'DA'
band_id = 'Beta'  # Frequency band
epoch_ids = ['DA', 'CF']  # Epoch identifier, defining 2 values on this list will compute connectivity for all epoch ids and then do the substraction
reject = False  # Peak to peak amplitude epoch rejection
data_type = 'ICA_annot'  # 'RAW' / 'ICA' / 'ICA_annot'

# Source estimation parameters
force_fsaverage = False  # Force all participants to use fsaverage regardless of their MRI data
model_name = 'lcmv'  # lcmv / dics
ico = 4  # Density of icasahedrons in source model (5, 6)
spacing = 10.  # Spacing of source model
surf_vol = 'surface'  # 'volume'/'surface'/'mixed'
pick_ori = None  # 'vector' For dipoles, 'max_power' for
if surf_vol == 'volume':
    parcelation_segmentation = 'aparc+aseg'  # aseg / aparc+aseg / aparc.a2009s+aseg
elif surf_vol == 'surface':
    parcelation_segmentation = 'aparc'  # aparc / aparc.a2009s

# Statistics parameters
correct_multiple_comparisons = True

# Connectivity parameters
compute_tmin = None
compute_tmax = None
if surf_vol == 'volume':
    labels_mode = 'mean'
elif surf_vol == 'surface':
    labels_mode = 'pca_flip'
envelope_connectivity = True
downsample_ts = True  # Downsample time series to desired_sfreq
desired_sfreq = 125  # Desired sampling frequency for envelope connectivity if downsample_ts is True
if envelope_connectivity:
    connectivity_method = 'corr'
    orthogonalization = 'pair'  # 'pair' for pairwise leakage correction / 'sym' for symmetric leakage correction
else:
    connectivity_method = 'pli'
standarize_normalize_con = 'norm'  # Standarize connectivity matrices within subjects


#----- Setup -----#
# Define Subjects_dir as Freesurfer output folder
subjects_dir = os.path.join(paths.mri_path, 'freesurfer')
os.environ["SUBJECTS_DIR"] = subjects_dir
# Get Source space for default subject
if surf_vol == 'volume':
    fname_src_default = paths.sources_path + 'fsaverage' + f'/fsaverage_volume_ico{ico}_{int(spacing)}-src.fif'
elif surf_vol == 'surface':
    fname_src_default = paths.sources_path + 'fsaverage' + f'/fsaverage_surface_ico{ico}-src.fif'
src_default = mne.read_source_spaces(fname_src_default)

# Path for envelope or signal connectivity
if envelope_connectivity:
    main_path = 'Connectivity_Env'
    # Modify path if downsample ts
    if downsample_ts:
        downsample_path = f'ds{desired_sfreq}'
    else:
        downsample_path = f'dsFalse'
    final_path = f'{orthogonalization}_{downsample_path}_{labels_mode}_{connectivity_method}'
else:
    main_path = 'Connectivity'
    final_path = f'{labels_mode}_{connectivity_method}'

# Turn on/off show figures
if display_figs:
    plt.ion()
else:
    plt.ioff()


#--------- Run ---------#
# Save data of each id
subj_matrices = {}
subj_matrices_no_std = {}
ga_matrices = {}
for epoch_id in epoch_ids:

    # Frequencies from band
    fmin, fmax = functions_general.get_freq_band(band_id=band_id)

    # Load data paths
    if envelope_connectivity:
        band_path = band_id
    elif not envelope_connectivity:
        band_path = 'None'

    # Run path
    run_path = f"Band_{band_path}/{epoch_id}_task{task}/"
    baseline_path = f"Band_{band_path}/baseline_task{task}/"

    # Epochs path
    epochs_save_path = paths.save_path + f"Epochs_{data_type}/{run_path}/"
    baseline_save_path = paths.save_path + f"Epochs_{data_type}/{baseline_path}/"

    # Source plots and data paths
    run_path_plot = run_path.replace('Band_None', f"Band_{band_id}")

    # Source estimation path
    if surf_vol == 'volume':
        source_model_path = f"{model_name}_{surf_vol}_ico{ico}_spacing{spacing}_{pick_ori}"
    elif surf_vol == 'surface':
        source_model_path = f"{model_name}_{surf_vol}_ico{ico}_{pick_ori}"
    labels_model_path = source_model_path + f"_{parcelation_segmentation}_{labels_mode}/"
    label_ts_save_path = paths.save_path + f"Source_labels_{data_type}/{run_path}/" + labels_model_path

    # Connectivity matrices plots and save paths
    fig_path = paths.plots_path + f"{main_path}_{data_type}/" + run_path_plot + source_model_path + f"_{parcelation_segmentation}_{final_path}_{standarize_normalize_con}/"
    save_path = paths.save_path + f"{main_path}_{data_type}/" + run_path_plot + source_model_path + f"_{parcelation_segmentation}_{final_path}/"

    # Save conectivity matrices
    subj_matrices[epoch_id] = []
    subj_matrices_no_std[epoch_id] = []
    ga_matrices[epoch_id] = []

    # Get parcelation labels and set up connectivity matrix
    if surf_vol == 'surface':  # or surf_vol == 'mixed':
        labels_fname = None
        # Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
        fsaverage_labels = mne.read_labels_from_annot(subject='fsaverage', parc=parcelation_segmentation, subjects_dir=subjects_dir)
        # Remove 'unknown' label for fsaverage aparc labels
        if parcelation_segmentation == 'aparc':
            print("Dropping extra 'unkown' label from lh.")
            drop_idxs = [i for i, label in enumerate(fsaverage_labels) if 'unknown' in label.name]
            for drop_idx in drop_idxs:
                fsaverage_labels.pop(drop_idx)
        # con_matrix = np.zeros((len(exp_info.subjects_ids), len(fsaverage_labels), len(fsaverage_labels)))

    elif surf_vol == 'volume':
        # fsaverage labels fname
        fsaverage_labels_fname = subjects_dir + f'/fsaverage/mri/{parcelation_segmentation}.mgz'
        # Get bem model
        fname_bem_fsaverage = paths.sources_path + 'fsaverage' + f'/fsaverage_bem_ico{ico}-sol.fif'
        bem_fsaverage = mne.read_bem_solution(fname_bem_fsaverage)

        # Get labels for FreeSurfer 'aseg' segmentation
        label_names_fsaverage = mne.get_volume_labels_from_aseg(fsaverage_labels_fname, return_colors=False)
        vol_labels_src_fsaverage = mne.setup_volume_source_space(subject='fsaverage', subjects_dir=subjects_dir, bem=bem_fsaverage, pos=spacing,
                                                       sphere_units='m', add_interpolator=True, volume_label=label_names_fsaverage)
        fsaverage_labels = mne.get_volume_labels_from_src(vol_labels_src_fsaverage, subject='fsaverage', subjects_dir=subjects_dir)

    con_matrix = []
    con_matrix_no_std = []
    # --------- Run ---------#
    for subj_num, subject_id in enumerate(exp_info.subjects_ids):

        # Load subject
        subject = setup.subject(subject_id=subject_id)

        data_tmin, data_tmax, _ = functions_general.get_time_lims(subject=subject, epoch_id=epoch_id)
        baseline_tmin, baseline_tmax, _ = functions_general.get_time_lims(subject=subject, epoch_id='baseline')

        # Baseline duration
        baseline, plot_baseline = functions_general.get_baseline_duration(epoch_id=epoch_id, tmin=data_tmin, tmax=data_tmin)

        # Connectivity tmin relative to epochs tmin time (label_ts restarts time to 0)
        if compute_tmin == None:
            con_tmin = 0
        else:
            con_tmin = compute_tmin - data_tmin
        if compute_tmax == None:
            con_tmax = data_tmax - data_tmin
        else:
            con_tmax = compute_tmax - data_tmin

        # --------- Coord systems alignment ---------#
        if force_fsaverage:
            subject_code = 'fsaverage'
            fs_subj_path = os.path.join(subjects_dir, subject_id)
            dig = False
        else:
            # Check if subject has MRI data
            try:
                fs_subj_path = os.path.join(subjects_dir, subject_id)
                os.listdir(fs_subj_path)
                dig = True
                subject_code = subject_id
            except:
                subject_code = 'fsaverage'
                fs_subj_path = os.path.join(subjects_dir, subject_code)
                dig = False

        # --------- Paths ---------#
        # Data filenames
        epochs_data_fname = f'Subject_{subject_id}_epo.fif'
        labels_ts_data_fname = f'Subject_{subject_id}.pkl'
        fname_lcmv = f'/{subject_code}_{data_type}_band{band_id}_{surf_vol}_ico{ico}_spacing{spacing}_{pick_ori}-lcmv.h5'

        # Save figures path
        fig_path_subj = fig_path + f'{subject_id}/'
        # Connectivity data fname
        fname_con = save_path + f'{subject_id}'

        # Source data path
        sources_path_subject = paths.sources_path + subject.subject_id
        # Load forward model
        if surf_vol == 'volume':
            fname_fwd = sources_path_subject + f'/{subject_code}_volume_ico{ico}_{int(spacing)}-fwd.fif'
        elif surf_vol == 'surface':
            fname_fwd = sources_path_subject + f'/{subject_code}_surface_ico{ico}-fwd.fif'
        fwd = mne.read_forward_solution(fname_fwd)
        # Get sources from forward model
        src = fwd['src']
        # Get bem model
        fname_bem = paths.sources_path + subject_code + f'/{subject_code}_bem_ico{ico}-sol.fif'
        bem = mne.read_bem_solution(fname_bem)

        # Parcellation labels
        if surf_vol == 'volume':
            labels_fname = subjects_dir + f'/{subject_code}/mri/{parcelation_segmentation}.mgz'
            # Get labels for FreeSurfer 'aseg' segmentation
            label_names = mne.get_volume_labels_from_aseg(labels_fname, return_colors=False)
            vol_labels_src = mne.setup_volume_source_space(subject=subject_code, subjects_dir=subjects_dir, bem=bem, pos=spacing,
                                                           sphere_units='m', add_interpolator=True, volume_label=label_names)
            labels = mne.get_volume_labels_from_src(vol_labels_src, subject=subject_code, subjects_dir=subjects_dir)

            label_names_segmentation = []
            for label in labels:
                if 'rh' in label.name and 'ctx' not in label.name:
                    label_name = 'Right-' + label.name.replace('-rh', '')
                elif 'lh' in label.name and 'ctx' not in label.name:
                    label_name = 'Left-' + label.name.replace('-lh', '')
                else:
                    label_name = label.name
                label_names_segmentation.append(label_name)

        elif subject_code != 'fsaverage':
            # Get labels for FreeSurfer cortical parcellation
            labels = mne.read_labels_from_annot(subject=subject_code, parc=parcelation_segmentation, subjects_dir=subjects_dir)
            label_names_segmentation = None
        else:
            labels = fsaverage_labels
            label_names_segmentation = None

        # Load connectivity matrix
        if os.path.isfile(fname_con):
            con = mne_connectivity.read_connectivity(fname_con)
        else:
            # Load labels ts data
            if os.path.isfile(label_ts_save_path + labels_ts_data_fname):
                label_ts = load.var(file_path=label_ts_save_path + labels_ts_data_fname)

                # Load MEG data
                if envelope_connectivity:
                    meg_data = load.meg(subject_id=subject_id, task=task, data_type=data_type, band_id=band_id)
                else:
                    meg_data = load.meg(subject_id=subject_id, task=task, data_type=data_type)
            else:
                # Load MEG data
                if envelope_connectivity:
                    meg_data = load.meg(subject_id=subject_id, task=task, data_type=data_type, band_id=band_id)
                else:
                    meg_data = load.meg(subject_id=subject_id, task=task, data_type=data_type)

                # Load epochs data
                if os.path.isfile(epochs_save_path + epochs_data_fname):
                    data_epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
                else:
                    # Epoch data
                    data_epochs, events = functions_analysis.epoch_data(subject=subject, epoch_id=epoch_id,
                                                                   meg_data=meg_data, tmin=data_tmin, tmax=data_tmax,
                                                                   baseline=baseline, save_data=save_data,
                                                                   epochs_save_path=epochs_save_path,
                                                                   epochs_data_fname=epochs_data_fname, reject=reject)
                data_epochs.pick('meg')

                if data_type == 'ICA_annot':
                    # --------- EXTRA ---------#
                    # 2. Get the epoch data and times
                    epoch_data = data_epochs.get_data()[0]  # Shape: (n_channels, n_times), single epoch
                    epoch_times = data_epochs.times  # Time vector relative to tmin

                    # 3. Use annotations to drop BAD segments
                    # Create a mask based on the annotations in the epoch
                    good_mask = np.ones(len(epoch_times), dtype=bool)
                    for annot in data_epochs.annotations:
                        if annot['description'] == 'BAD':  # Only consider BAD annotations
                            start = max(annot['onset'] - data_epochs.tmin, 0)  # Adjust to epoch time
                            end = min(annot['onset'] + annot['duration'] - data_epochs.tmin, data_tmax - data_tmin)
                            bad_indices = (epoch_times >= start) & (epoch_times <= end)
                            good_mask[bad_indices] = False

                    # 4. Apply the mask to keep only good segments
                    good_data = epoch_data[:, good_mask]  # Shape: (n_channels, n_good_times)
                    good_times = epoch_times[good_mask]

                    # 5. (Optional) Create a new Raw object with the cleaned data
                    info = data_epochs.info.copy()
                    cleaned_raw = mne.io.RawArray(good_data, info)

                    # Create epochs from the cleaned Raw object
                    data_epochs = mne.Epochs(cleaned_raw, np.array([[0, 0, 1]]), event_id={'condition': 1}, tmin=0,
                                             tmax=cleaned_raw.times[-1], baseline=None, preload=True)

                # --------- Source estimation ---------#
                # Define filter
                if os.path.isfile(sources_path_subject + fname_lcmv):
                    filters = mne.beamformer.read_beamformer(sources_path_subject + fname_lcmv)
                else:
                    meg_data.pick('meg')
                    data_cov = mne.compute_raw_covariance(meg_data)
                    filters = beamformer.make_lcmv(info=meg_data.info, forward=fwd, data_cov=data_cov, reg=0.05, pick_ori=pick_ori)
                    filters.save(fname=sources_path_subject + fname_lcmv, overwrite=True)

                # Apply filter and get source estimates
                stc_epochs = beamformer.apply_lcmv_epochs(epochs=data_epochs, filters=filters, return_generator=True)

                # Average the source estimates within each label using sign-flips to reduce signal cancellations
                if surf_vol == 'volume':
                    label_ts = mne.extract_label_time_course(stcs=stc_epochs, labels=(labels_fname, label_names_segmentation), src=src, mode=labels_mode,
                                                             return_generator=False)
                elif surf_vol == 'surface':
                    label_names_segmentation = None
                    label_ts = mne.extract_label_time_course(stcs=stc_epochs, labels=labels, src=src, mode=labels_mode, return_generator=False)

                # Save
                if save_data:
                    os.makedirs(label_ts_save_path, exist_ok=True)
                    save.var(var=label_ts, path=label_ts_save_path, fname=labels_ts_data_fname)

            if envelope_connectivity:
                if downsample_ts:
                    for i, ts in enumerate(label_ts):
                        sfreq = meg_data.info['sfreq']
                        samples_interval = int(sfreq/desired_sfreq)
                        # Taking jumping windows average of samples
                        label_ts[i] = np.array([np.mean(ts[:, j*samples_interval:(j+1)*samples_interval], axis=-1) for j in range(int(len(ts[0])/samples_interval))]).T
                        # Subsampling
                        # label_ts[i] = ts[:, ::samples_interval]

                # Compute envelope connectivity (automatically computes hilbert transform to extract envelope)
                if orthogonalization == 'pair':
                    con = mne_connectivity.envelope_correlation(data=label_ts, names=[label.name for label in labels])

                elif orthogonalization == ' sym':
                    label_ts_orth = mne_connectivity.envelope.symmetric_orth(label_ts)
                    con = mne_connectivity.envelope_correlation(label_ts_orth, orthogonalize=False)
                    # Take absolute value of correlations (orthogonalize False does not take abs by default)
                    con.xarray.data = abs(con.get_data())

                # Average across epochs
                con = con.combine()

            else:
                con = mne_connectivity.spectral_connectivity_epochs(label_ts, method=connectivity_method, mode='multitaper', sfreq=meg_data.info['sfreq'],
                                                                    fmin=fmin, fmax=fmax, tmin=con_tmin, tmax=con_tmax, faverage=True, mt_adaptive=True)
            # Save
            if save_data:
                os.makedirs(save_path, exist_ok=True)
                con.save(fname_con)

        # Get connectivity matrix
        con_subj = con.get_data(output='dense')[:, :, 0]
        con_subj = np.maximum(con_subj, con_subj.transpose())  # make symetric

        # Save for comparisons
        con_matrix_no_std.append(con_subj)

        # Standarize
        if standarize_normalize_con == 'std':
            con_subj = (con_subj - np.mean(con_subj)) / np.std(con_subj)
        # Normalize
        elif standarize_normalize_con == 'norm':
            con_subj = con_subj / np.sqrt(np.mean(con_subj**2))

        # Save for GA
        con_matrix.append(con_subj)

        if plot_individuals:

            # Plot circle
            plot_general.connectivity_circle(subject=subject, labels=labels, surf_vol=surf_vol, con=con_subj, connectivity_method=connectivity_method,
                                             subject_code=subject_code, display_figs=display_figs, save_fig=save_fig, fig_path=fig_path_subj, fname=None)

            # Plot connectome
            plot_general.connectome(subject=subject, labels=labels, adjacency_matrix=con_subj, subject_code=subject_code, save_fig=save_fig,
                                    fig_path=fig_path_subj, fname=None)

            # Plot connectivity matrix
            sorted_matrix = plot_general.plot_con_matrix(subject=subject, labels=labels, adjacency_matrix=con_subj, subject_code=subject_code,
                                         save_fig=save_fig, fig_path=fig_path_subj, fname=None)

            # Plot connectivity strength (connections from each region to other regions)
            plot_general.connectivity_strength(subject=subject, subject_code=subject_code, con=con, src=src, labels=labels, surf_vol=surf_vol, labels_fname=labels_fname,
                                               label_names_segmentation=label_names_segmentation, subjects_dir=subjects_dir, save_fig=save_fig, fig_path=fig_path_subj, fname=None)

    # --------- Grand Average ---------#
    # Get connectivity matrix for GA
    ga_con_matrix = np.array(con_matrix).mean(0)
    # Fill diagonal with 0
    np.fill_diagonal(ga_con_matrix, 0)

    # Plot circle
    plot_general.connectivity_circle(subject='GA', labels=fsaverage_labels, surf_vol=surf_vol, con=ga_con_matrix, connectivity_method=connectivity_method, subject_code='fsaverage',
                                     display_figs=display_figs, save_fig=save_fig, fig_path=fig_path, fname='GA_circle')

    # Plot connectome
    plot_general.connectome(subject='GA', labels=fsaverage_labels, adjacency_matrix=ga_con_matrix, subject_code='fsaverage', save_fig=save_fig, fig_path=fig_path,
                            fname='GA_connectome')

    # Plot matrix
    ga_sorted_matrix = plot_general.plot_con_matrix(subject='GA', labels=fsaverage_labels, adjacency_matrix=ga_con_matrix, subject_code='fsaverage',
                                                    save_fig=save_fig, fig_path=fig_path, fname='GA_matrix')

    # Plot connectivity strength (connections from each region to other regions)
    plot_general.connectivity_strength(subject='GA', subject_code='fsaverage', con=ga_con_matrix, src=src_default, labels=fsaverage_labels, surf_vol=surf_vol,
                                       labels_fname=labels_fname, label_names_segmentation=label_names_segmentation, subjects_dir=subjects_dir, save_fig=save_fig,
                                       fig_path=fig_path, fname='GA_strength')

    # Get connectivity matrices for comparisson
    subj_matrices[epoch_id] = np.array(con_matrix)
    subj_matrices_no_std[epoch_id] = np.array(con_matrix_no_std)
    ga_matrices[epoch_id] = ga_sorted_matrix


# ----- Difference between conditions ----- #
# Take difference of conditions if applies
if len(epoch_ids) > 1:
    for comparison in list(itertools.combinations(epoch_ids, 2)):
    # for epoch_id in epoch_ids:
        # Redefine figure save path
        fig_path_diff = fig_path.replace(f'{epoch_ids[-1]}', f'{comparison[0]}-{comparison[1]}')

        print(f'Comparing conditions {comparison[0]} - {comparison[1]}')

        #------ RSA ------#
        # Compute RSA between GA matrices from both conditions
        rsa_result = mne_rsa.rsa(ga_matrices[comparison[0]], ga_matrices[comparison[1]], metric="spearman")
        # Plot Connectivity matrices from both conditions
        fig = mne_rsa.plot_rdms([ga_matrices[comparison[0]], ga_matrices[comparison[1]]], names=[comparison[0], comparison[1]])
        fig.suptitle(f'RSA: {round(rsa_result, 2)}')

        # Save
        if save_fig:
            fname = f'GA_rsa'
            save.fig(fig=fig, path=fig_path_diff, fname=fname)

        #------ t-test ------#
        # Connectivity t-values variable
        t_values, p_values = wilcoxon(x=subj_matrices[comparison[0]], y=subj_matrices[comparison[1]], axis=0)

        # Significance thresholds
        p_threshold = 0.05
        ravel_p_values = p_values.ravel()

        # Make 1D arrays to run FDR correction
        if correct_multiple_comparisons:
            rejected, corrected_pval = fdrcorrection(pvals=ravel_p_values, alpha=p_threshold)  # rejected refers to null hypothesis
        else:
            rejected = ravel_p_values < p_threshold
            corrected_pval = ravel_p_values

        # Reshape to regions x regions array
        corrected_pval = np.reshape(corrected_pval, newshape=p_values.shape)
        rejected = np.reshape(rejected, newshape=p_values.shape)

        # Take significant links (in case asymetric results)
        rejected = np.maximum(rejected, rejected.transpose())

        # Discard diagonal
        np.fill_diagonal(rejected, False)

        # Mask p-values by significance
        corrected_pval[~rejected.astype(bool)] = 1
        log_p_values = -np.log10(corrected_pval)

        # Mask t-values by significance
        t_values[~rejected] = 0

        # Plot significant t-values
        if t_values.any() > 0:
            min_value = sorted(set(np.sort(t_values, axis=None)))[0]/2
            max_value = sorted(set(np.sort(t_values, axis=None)))[-1]

            # Plot circle
            plot_general.connectivity_circle(subject='GA', labels=labels, surf_vol=surf_vol, con=t_values, connectivity_method=connectivity_method,
                                             vmin=min_value, vmax=max_value, subject_code='fsaverage', display_figs=display_figs, save_fig=save_fig, fig_path=fig_path_diff,
                                             fname='GA_circle_t')

            # Plot p-values connectome
            plot_general.connectome(subject='GA', labels=labels, adjacency_matrix=t_values, subject_code='fsaverage',
                                    save_fig=save_fig, fig_path=fig_path_diff, fname=f'GA_t_con', connections_num=(log_p_values > 0).sum())

            # Plot matrix
            sig_diff_sorted_matrix = plot_general.plot_con_matrix(subject='GA', labels=labels, adjacency_matrix=t_values, subject_code='fsaverage',
                                         save_fig=save_fig, fig_path=fig_path_diff, fname='GA_matrix_t')

            # Plot connectivity strength (connections from each region to other regions)
            plot_general.connectivity_strength(subject='GA', subject_code='fsaverage', con=t_values, src=src_default, labels=fsaverage_labels, surf_vol=surf_vol,
                                               labels_fname=labels_fname, label_names_segmentation=label_names_segmentation,
                                               subjects_dir=subjects_dir, save_fig=save_fig, fig_path=fig_path_diff, fname=f'GA_strength_t')

        #----- Difference -----#
        con_diff = []
        mean_global_con = {comparison[0]: [], comparison[1]: []}
        mean_global_con_diff = []
        # Compute difference for cross2
        for i in range(len(subj_matrices_no_std[comparison[0]])):
            # Global mean connectivity
            mean_global_con[comparison[0]].append(subj_matrices_no_std[comparison[0]][i].mean())
            mean_global_con[comparison[1]].append(subj_matrices_no_std[comparison[1]][i].mean())
            mean_global_con_diff.append((mean_global_con[comparison[0]][i] - mean_global_con[comparison[1]][i]) /
                                           (mean_global_con[comparison[0]][i] + mean_global_con[comparison[1]][i]))
            # Matrix differences
            subj_dif = subj_matrices_no_std[comparison[0]][i] - subj_matrices_no_std[comparison[1]][i]
            if standarize_normalize_con == 'std':
                subj_dif = (subj_dif - np.mean(subj_dif)) / np.std(subj_dif)
            elif standarize_normalize_con == 'norm':
                subj_dif = subj_dif / np.sqrt(np.mean(subj_dif**2))
            con_diff.append(subj_dif)

        # Make array
        con_diff = np.array(con_diff)

        # Take Grand Average of connectivity differences
        con_diff_ga = con_diff.mean(0)

        # Fill diagonal with 0
        np.fill_diagonal(con_diff_ga, 0)

        # Plot global mean connectivity
        plot_general.global_mean_con(subject='GA', mean_global_con=mean_global_con, subject_code='fsaverage',
                                     save_fig=save_fig, fig_path=fig_path_diff, fname='GA_global_mean')

        # Plot circle
        plot_general.connectivity_circle(subject='GA', labels=labels, surf_vol=surf_vol, con=con_diff_ga, connectivity_method=connectivity_method, subject_code='fsaverage',
                                         display_figs=display_figs, save_fig=save_fig, fig_path=fig_path_diff, fname='GA_circle_dif')

        # Plot connectome
        plot_general.connectome(subject='GA', labels=labels, adjacency_matrix=con_diff_ga, subject_code='fsaverage', edge_thresholddirection='absabove',
                                save_fig=save_fig, fig_path=fig_path_diff, fname='GA_connectome_dif')

        # Plot matrix
        diff_sorted_matrix = plot_general.plot_con_matrix(subject='GA', labels=labels, adjacency_matrix=con_diff_ga, subject_code='fsaverage',
                                     save_fig=save_fig, fig_path=fig_path_diff, fname='GA_matrix_dif')

        # Plot connectivity strength (connections from each region to other regions)
        plot_general.connectivity_strength(subject='GA', subject_code='fsaverage', con=con_diff_ga, src=src_default, labels=fsaverage_labels, surf_vol=surf_vol,
                                           labels_fname=labels_fname, label_names_segmentation=label_names_segmentation, subjects_dir=subjects_dir, save_fig=save_fig,
                                           fig_path=fig_path_diff, fname='GA_strength_dif')