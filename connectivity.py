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
from scipy.stats import ttest_ind, wilcoxon
from statsmodels.stats.multitest import fdrcorrection

# Removing NANs from meg_data
# meg_data.crop(tmin=0, tmax=meg_data.times[np.where(~np.isnan(meg_data._data[0]))[-1]][0])
# meg_data.save(paths.ica_path + f'{subject_id}/' + f'{task}_3_raw_ica_hfc_meg.fif', overwrite=True)

# Load experiment info
exp_info = setup.exp_info()

# --------- Define Parameters ---------#
save_fig = True
display_figs = True
plot_individuals = True
save_data = True

# Trial parameters
task = 'DA'
band_id = 'HGamma'  # Frequency band
epoch_ids = ['DA', 'CF']  # Epoch identifier, defining 2 values on this list will compute connectivity for all epoch ids and then do the substraction
reject = False  # Peak to peak amplitude epoch rejection
data_type = 'ICA'  # 'RAW'

# Source estimation parameters
force_fsaverage = False  # Force all participants to use fsaverage regardless of their MRI data
model_name = 'lcmv'  # lcmv / dics
ico = 4  # Density of icasahedrons in source model (5, 6)
surf_vol = 'surface'  # 'volume'/'surface'/'mixed'
pick_ori = None  # 'vector' For dipoles, 'max_power' for
parcelation = 'aparc'  # aparc / aparc.a2009s

# Statistics parameters
correct_multiple_comparisons = True

# Connectivity parameters
compute_tmin = None
compute_tmax = None
labels_mode = 'pca_flip'
envelope_connectivity = True
if envelope_connectivity:
    connectivity_method = 'corr'
    orthogonalization = 'pair'  # 'pair' for pairwise leakage correction / 'sym' for symmetric leakage correction
    downsample_ts = False
    desired_sfreq = 10
else:
    connectivity_method = 'pli'
standarize_con = True  # Standarize connectivity matrices within subjects


#----- Setup -----#
# Define Subjects_dir as Freesurfer output folder
subjects_dir = os.path.join(paths.mri_path, 'freesurfer')
os.environ["SUBJECTS_DIR"] = subjects_dir
# Get Source space for default subject
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
    source_model_path = f"{model_name}_{surf_vol}_ico{ico}_{pick_ori}"
    labels_model_path = source_model_path + f"_{parcelation}_{labels_mode}/"
    label_ts_save_path = paths.save_path + f"Source_labels_{data_type}/{run_path}/" + labels_model_path

    # Connectivity matrices plots and save paths
    fig_path = paths.plots_path + f"{main_path}_{data_type}/" + run_path_plot + source_model_path + f"_{parcelation}_{final_path}_std{standarize_con}/"
    save_path = paths.save_path + f"{main_path}_{data_type}/" + run_path_plot + source_model_path + f"_{parcelation}_{final_path}/"

    # Save conectivity matrices
    subj_matrices[epoch_id] = []
    ga_matrices[epoch_id] = []

    # Get parcelation labels and set up connectivity matrix
    # Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
    fsaverage_labels = mne.read_labels_from_annot(subject='fsaverage', parc=parcelation, subjects_dir=subjects_dir)
    # Remove 'unknown' label for fsaverage aparc labels
    if parcelation == 'aparc':
        print("Dropping extra 'unkown' label from lh.")
        drop_idxs = [i for i, label in enumerate(fsaverage_labels) if 'unknown' in label.name]
        for drop_idx in drop_idxs:
            fsaverage_labels.pop(drop_idx)
    con_matrix = np.zeros((len(exp_info.subjects_ids), len(fsaverage_labels), len(fsaverage_labels)))

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
        fname_lcmv = f'/{subject_code}_band{band_id}_{surf_vol}_ico{ico}_{pick_ori}-lcmv.h5'

        # Save figures path
        fig_path_subj = fig_path + f'{subject_id}/'
        # Connectivity data fname
        fname_con = save_path + f'{subject_id}'

        # Source data path
        sources_path_subject = paths.sources_path + subject.subject_id
        # Load forward model
        fname_fwd = sources_path_subject + f'/{subject_code}_{surf_vol}_ico{ico}-fwd.fif'
        fwd = mne.read_forward_solution(fname_fwd)
        # Get sources from forward model
        src = fwd['src']

        # Parcellation labels
        if subject_code != 'fsaverage':
            # Get labels for FreeSurfer cortical parcellation
            labels = mne.read_labels_from_annot(subject=subject_code, parc=parcelation, subjects_dir=subjects_dir)
        else:
            labels = fsaverage_labels

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
                label_ts = mne.extract_label_time_course(stcs=stc_epochs, labels=labels, src=src, mode=labels_mode, return_generator=False)

                # Save
                if save_data:
                    os.makedirs(save_path, exist_ok=True)
                    save.var(var=label_ts, path=label_ts_save_path, fname=labels_ts_data_fname)

            if envelope_connectivity:
                if downsample_ts:
                    for i, ts in enumerate(label_ts):
                        sfreq = data_epochs.info['sfreq']
                        samples_interval = int(sfreq/desired_sfreq)
                        # Taking jumping windows average of samples
                        label_ts[i] = np.array([np.mean(ts[:, j*samples_interval:(j+1)*samples_interval], axis=-1) for j in range(int(len(ts[0])/samples_interval) + 1)]).T
                        # Subsampling
                        # label_ts[i] = ts[:, ::samples_interval]

                # Compute envelope connectivity (automatically computes hilbert transform to extract envelope)
                if orthogonalization == 'pair':
                    label_names = [label.name for label in labels]
                    con = mne_connectivity.envelope_correlation(data=label_ts, names=label_names)

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

        # Standarize
        if standarize_con:
            con_subj = (con_subj - np.mean(con_subj)) / np.std(con_subj)

        # Save for GA
        con_matrix[subj_num] = con_subj

        if plot_individuals:
            # Plot circle
            plot_general.connectivity_circle(subject=subject, labels=labels, surf_vol=surf_vol, con=con_matrix[subj_num], connectivity_method=connectivity_method,
                                             subject_code=subject_code, display_figs=display_figs, save_fig=save_fig, fig_path=fig_path_subj, fname=None)

            # Plot connectome
            plot_general.connectome(subject=subject, labels=labels, adjacency_matrix=con_matrix[subj_num], subject_code=subject_code,
                                    save_fig=save_fig, fig_path=fig_path_subj, fname=None)

            # Plot connectivity matrix
            plot_general.plot_con_matrix(subject=subject, labels=labels, adjacency_matrix=con_matrix[subj_num], subject_code=subject_code,
                                         save_fig=save_fig, fig_path=fig_path_subj, fname=None)

            # Plot connectivity strength (connections from each region to other regions)
            plot_general.connectivity_strength(subject=subject, subject_code=subject_code, con=con, src=src, labels=labels, surf_vol=surf_vol,
                                               subjects_dir=subjects_dir, save_fig=save_fig, fig_path=fig_path_subj, fname=None)

    # --------- Grand Average ---------#
    # Get connectivity matrix for GA
    ga_con_matrix = con_matrix.mean(0)
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
                                       subjects_dir=subjects_dir, save_fig=save_fig, fig_path=fig_path, fname='GA_strength')

    # Get connectivity matrices for comparisson
    subj_matrices[epoch_id] = con_matrix
    ga_matrices[epoch_id] = ga_sorted_matrix


# ----- Difference between conditions ----- #
# Take difference of conditions if applies
if len(epoch_ids) > 1:
    for epoch_id in epoch_ids:
        # Redefine figure save path
        fig_path_diff = fig_path.replace(f'{epoch_ids[-1]}', f'{epoch_ids[0]}-{epoch_ids[1]}')

        print(f'Comparing conditions {epoch_ids[0]} - {epoch_ids[1]}')

        #------ RSA ------#
        # Compute RSA between GA matrices from both conditions
        rsa_result = mne_rsa.rsa(ga_matrices[epoch_ids[0]], ga_matrices[epoch_ids[1]], metric="spearman")
        # Plot Connectivity matrices from both conditions
        fig = mne_rsa.plot_rdms([ga_matrices[epoch_ids[0]], ga_matrices[epoch_ids[1]]], names=[epoch_ids[0], epoch_ids[1]])
        fig.suptitle(f'RSA: {round(rsa_result, 2)}')

        # Save
        if save_fig:
            fname = f'GA_rsa'
            save.fig(fig=fig, path=fig_path_diff, fname=fname)

        #------ t-test ------#
        # Connectivity t-values variable
        t_values, p_values = wilcoxon(x=subj_matrices[epoch_ids[0]], y=subj_matrices[epoch_ids[1]], axis=0)

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

            # Plot matrix
            plot_general.plot_con_matrix(subject='GA', labels=fsaverage_labels, adjacency_matrix=t_values, subject_code='fsaverage',
                                         save_fig=save_fig, fig_path=fig_path_diff, fname='GA_matrix_t')

            # Plot circle
            plot_general.connectivity_circle(subject='GA', labels=fsaverage_labels, surf_vol=surf_vol, con=t_values, connectivity_method=connectivity_method, vmin=min_value,
                                             vmax=max_value, subject_code='fsaverage', display_figs=display_figs, save_fig=save_fig, fig_path=fig_path_diff, fname='GA_circle_t')

            # Plot p-values connectome
            plot_general.connectome(subject='GA', labels=fsaverage_labels, adjacency_matrix=t_values, subject_code='fsaverage', save_fig=save_fig, fig_path=fig_path_diff,
                                    fname=f'GA_t_con', cmap='Reds', edge_vmin=min_value, edge_vmax=max_value, connections_num=(log_p_values > 0).sum())

            # Plot connectivity strength (connections from each region to other regions)
            plot_general.connectivity_strength(subject='GA', subject_code='fsaverage', con=t_values, src=src_default, labels=fsaverage_labels, surf_vol=surf_vol,
                                               subjects_dir=subjects_dir, save_fig=save_fig, fig_path=fig_path_diff, fname=f'GA_strength_t')

        #----- Difference -----#
        con_diff = []
        # Compute difference for cross2
        for i in range(len(subj_matrices[epoch_ids[0]])):
            con_diff.append(subj_matrices[epoch_ids[0]][i] - subj_matrices[epoch_ids[1]][i])

        # Make array
        con_diff = np.array(con_diff)

        # Take Grand Average of connectivity differences
        con_diff_ga = con_diff.mean(0)

        # Fill diagonal with 0
        np.fill_diagonal(con_diff_ga, 0)

        # Plot circle
        plot_general.connectivity_circle(subject='GA', labels=fsaverage_labels, surf_vol=surf_vol, con=con_diff_ga, connectivity_method=connectivity_method, subject_code='fsaverage',
                                         display_figs=display_figs, save_fig=save_fig, fig_path=fig_path_diff, fname='GA_circle_dif')

        # Plot connectome
        plot_general.connectome(subject='GA', labels=fsaverage_labels, adjacency_matrix=con_diff_ga, subject_code='fsaverage',
                                save_fig=save_fig, fig_path=fig_path_diff, fname='GA_connectome_dif')

        # Plot matrix
        plot_general.plot_con_matrix(subject='GA', labels=fsaverage_labels, adjacency_matrix=con_diff_ga, subject_code='fsaverage',
                                     save_fig=save_fig, fig_path=fig_path_diff, fname='GA_matrix_dif')

        # Plot connectivity strength (connections from each region to other regions)
        plot_general.connectivity_strength(subject='GA', subject_code='fsaverage', con=con_diff_ga, src=src_default, labels=fsaverage_labels, surf_vol=surf_vol,
                                           subjects_dir=subjects_dir, save_fig=save_fig, fig_path=fig_path_diff, fname='GA_strength_dif')