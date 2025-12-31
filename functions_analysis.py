import mne
import numpy as np
import pandas as pd
import os
import setup
from setup import exp_info
import paths
import load
import save
import functions_general
from mne.decoding import ReceptiveField
from scipy import stats as stats
from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc
from mni_to_atlas import AtlasBrowser


exp_info = setup.exp_info()

# ---------- Epoch Data ---------- #
def define_events(subject, meg_data, epoch_id, epoch_keys=None):
    """
    Define events based on annotations in MEG data.

    Parameters
    ----------
    subject : instance of subect class defined in setup.py
        The object containing all subject information and parameters.
    meg_data : instance of mne.io.Raw
        The raw MEG data.
    epoch_id : str
        The identifier for the epoch, which can include multiple sub-ids separated by '+'.
    epoch_keys : list of str, optional
        Specific event keys to use. If None, all events matching the epoch_id will be used.

    Returns
    -------
    metadata : pandas.DataFrame
        Metadata for the epochs.
    events : array, shape (n_events, 3)
        The events array.
    events_id : dict
        The dictionary of event IDs.

    Raises
    ------
    ValueError
        If no valid epoch_ids are provided.
    """

    print('Defining events')
    onset_times = None
    if 'fix' in epoch_id or 'sac' in epoch_id or 'pur' in epoch_id:
        if 'fix' in epoch_id:
            # Load df of events
            metadata = subject.fixations()

        if 'sac' in epoch_id:
            # Load df of events
            metadata = subject.saccades()

        if 'pur' in epoch_id:
            # Load df of events
            metadata = subject.pursuits()

        events = np.zeros((len(metadata), 3))

        # Make events array
        events[:, 0] = round((metadata['onset'] + meg_data.first_time) * meg_data.info['sfreq'], 0)  # Convert seconds to samples
        events[:, 2] = 1
        events = events.astype(int)
        event_id = {np.str_(epoch_id): 1}

    else:
        if epoch_keys is None:

            if 'CF' == epoch_id:
                # Get task onset times
                drive_onset_time = meg_data.annotations.onset[np.where(meg_data.annotations.description == 'drive')[0]][0]
                onset_times = [subject.exp_times['cf_start'] + drive_onset_time - meg_data.first_time]
                onset_description = ['CF_onset'] * len(onset_times)
                task_duration = [exp_info.DA_duration] * len(onset_times)

                # Add annotations to MEG data
                stim_annotations = mne.Annotations(onset=onset_times,
                                                   duration=task_duration,
                                                   description=onset_description
                                                   )

                meg_data_copy = meg_data.copy()
                meg_data_copy.set_annotations(stim_annotations)

                # Get events from annotations
                events, event_id = mne.events_from_annotations(meg_data_copy, verbose=False)

                # Define description to epoch data
                epoch_keys = ['CF_onset']

            elif 'DA1' == epoch_id:
                # Get task onset times as Excel times + 'drive' annotation time (Joaco's decision CHECK PLEASE) Changed to one long epoch
                # drive_onset_time = meg_data.annotations.onset[np.where(meg_data.annotations.description == 'drive')[0]][0]
                # onset_times = [time + drive_onset_time for time in subject.da_times['DA times']]
                drive_onset_time = meg_data.annotations.onset[np.where(meg_data.annotations.description == 'drive')[0]][0]
                onset_times = [subject.da_times['DA times'][0] + drive_onset_time - meg_data.first_time]

                onset_description = ['DA_onset'] * len(onset_times)
                task_duration = [exp_info.DA_duration] * len(onset_times)

                # Add annotations to MEG data
                stim_annotations = mne.Annotations(onset=onset_times,
                                                   duration=task_duration,
                                                   description=onset_description
                                                   )

                meg_data_copy = meg_data.copy()
                meg_data_copy.set_annotations(stim_annotations)

                # Get events from annotations
                events, event_id = mne.events_from_annotations(meg_data_copy, verbose=False)

                # Define description to epoch data
                epoch_keys = ['DA_onset']

            elif 'DAall' == epoch_id:
                onset_times = np.array([subject.master_df['symbol_onset_time'] - meg_data.first_time]).squeeze()

                onset_description = ['DAall'] * len(onset_times)
                task_duration = [exp_info.DA_duration] * len(onset_times)

                # Add annotations to MEG data
                stim_annotations = mne.Annotations(onset=onset_times,
                                                   duration=task_duration,
                                                   description=onset_description
                                                   )

                meg_data_copy = meg_data.copy()
                meg_data_copy.set_annotations(stim_annotations)

                # Get events from annotations
                events, event_id = mne.events_from_annotations(meg_data_copy, verbose=False)

                # Define description to epoch data
                epoch_keys = ['DAall']

            elif 'DAfull' == epoch_id:
                drive_onset_time = meg_data.annotations.onset[np.where(meg_data.annotations.description == 'drive')[0]][0]
                onset_times = [subject.da_times['DA times'] + drive_onset_time - meg_data.first_time]

                # Create events for all time points during the DA duration
                sfreq = meg_data.info['sfreq']
                events_list = []

                for onset in onset_times:
                    # Convert onset time to sample index
                    onset_sample = int(onset * sfreq)
                    # Convert duration to number of samples
                    duration_samples = int(exp_info.DA_duration * sfreq)
                    # Create events for all samples in the duration
                    for sample_offset in range(duration_samples):
                        events_list.append([onset_sample + sample_offset, 0, 1])

                events = np.array(events_list)
                event_id = {'DAfull': 1}

                # Define description to epoch data
                epoch_keys = ['DAfull']

            elif 'left_but' == epoch_id:
                master_df = subject.master_df.loc[subject.master_df['resp_label'] == 1]
                onset_times = np.array([master_df['symbol_onset_time'] + master_df['reaction_time'] - meg_data.first_time]).squeeze()

                onset_description = ['left_but'] * len(master_df)
                task_duration = [0] * len(onset_times)

                # Add annotations to MEG data
                stim_annotations = mne.Annotations(onset=onset_times,
                                                   duration=task_duration,
                                                   description=onset_description
                                                   )

                meg_data_copy = meg_data.copy()
                meg_data_copy.set_annotations(stim_annotations)

                # Get events from annotations
                events, event_id = mne.events_from_annotations(meg_data_copy, verbose=False)

                # Define description to epoch data
                epoch_keys = ['left_but']

            elif 'right_but' == epoch_id:
                master_df = subject.master_df.loc[subject.master_df['resp_label'] == 4]
                onset_times = np.array([master_df['symbol_onset_time'] + master_df['reaction_time'] - meg_data.first_time]).squeeze()

                onset_description = ['right_but'] * len(master_df)
                task_duration = [0] * len(onset_times)

                # Add annotations to MEG data
                stim_annotations = mne.Annotations(onset=onset_times,
                                                   duration=task_duration,
                                                   description=onset_description
                                                   )

                meg_data_copy = meg_data.copy()
                meg_data_copy.set_annotations(stim_annotations)

                # Get events from annotations
                events, event_id = mne.events_from_annotations(meg_data_copy, verbose=False)

                # Define description to epoch data
                epoch_keys = ['right_but']

            elif 'baseline' == epoch_id:  # Baseline is from drive start to cf start
                # Get task onset times as Excel times + 'drive' annotation time
                drive_onset_time = meg_data.annotations.onset[np.where(meg_data.annotations.description == 'drive')[0]][0]
                onset_times = [drive_onset_time  - meg_data.first_time]

                onset_description = ['drive_onset'] * len(onset_times)
                task_duration = [subject.exp_times['cf_start']] * len(onset_times)  # cf time in excel file is relative to drive start.

                # Add annotations to MEG data
                stim_annotations = mne.Annotations(onset=onset_times,
                                                   duration=task_duration,
                                                   description=onset_description
                                                   )

                meg_data_copy = meg_data.copy()
                meg_data_copy.set_annotations(stim_annotations)

                # Get events from annotations
                events, event_id = mne.events_from_annotations(meg_data_copy, verbose=False)

                # Define description to epoch data
                epoch_keys = ['drive_onset']

            else:
                # Get events from annotations
                events, event_id = mne.events_from_annotations(meg_data, verbose=False)
                # events[:, 0] = events[:, 0] - functions_general.find_nearest(meg_data.times, meg_data.first_time)[0]  # Adjust event times to start from 0

                # Get epoch keys from epoch_id
                epoch_keys = epoch_id.split('+')

        else:
            # Get events from annotations
            events, event_id = mne.events_from_annotations(meg_data, verbose=False)

        # Get events and ids matching selection
        metadata, events, event_id = mne.epochs.make_metadata(events=events, event_id=event_id, row_events=epoch_keys, tmin=0, tmax=0, sfreq=meg_data.info['sfreq'])

    return metadata, events, event_id, onset_times


# def define_events_from_df(subject, meg_data, epoch_id, epoch_keys=None):
#     """
#     Define events based on annotations in MEG data.
#
#     Parameters
#     ----------
#     subject : instance of subect class defined in setup.py
#         The object containing all subject information and parameters.
#     meg_data : instance of mne.io.Raw
#         The raw MEG data.
#     epoch_id : str
#         The identifier for the epoch, which can include multiple sub-ids separated by '+'.
#     epoch_keys : list of str, optional
#         Specific event keys to use. If None, all events matching the epoch_id will be used.
#
#     Returns
#     -------
#     metadata : pandas.DataFrame
#         Metadata for the epochs.
#     events : array, shape (n_events, 3)
#         The events array.
#     events_id : dict
#         The dictionary of event IDs.
#
#     Raises
#     ------
#     ValueError
#         If no valid epoch_ids are provided.
#     """
#
#     print('Defining events')
#     onset_times = None
#     # Get events from annotations
#     events, event_id = mne.events_from_annotations(meg_data, verbose=False)
#
#     if 'fix' in epoch_id:
#         # Load df of events
#         metadata = subject.fixations()
#
#     if 'sac' in epoch_id:
#         # Load df of events
#         metadata = subject.saccades()
#
#     if 'pur' in epoch_id:
#         # Load df of events
#         metadata = subject.pursuits()
#
#     events = np.zeros((len(metadata), 3))
#
#     # Make events array
#     events[:, 0] = round((metadata['onset'] + meg_data.first_time) * meg_data.info['sfreq'], 0)  # Convert seconds to samples
#     events[:, 2] = 1
#     events = events.astype(int)
#
#     event_id = {np.str_(epoch_id): 1}
#
#     return metadata, events, event_id, onset_times


def epoch_data(subject, epoch_id, meg_data, tmin, tmax, from_df, baseline=(0, 0), reject=None, save_data=False, epochs_save_path=None, epochs_data_fname=None):
    """
    Epoch the MEG data based on the provided parameters.

    Parameters
    ----------
    subject : instance of subect class defined in setup.py
        The object containing all subject information and parameters.
    epoch_id : str
     The identifier for the epoch, which can include multiple sub-ids separated by '+'.
    meg_data : instance of mne.io.Raw
     The raw MEG data.
    tmin : float
     Start time before event.
    tmax : float
     End time after event.
    baseline : tuple of (float, float) or None, optional
     The time interval to apply baseline correction. Default is (None, 0).
    reject : dict or None, optional
     Rejection parameters based on channel amplitude. Default is None.
    save_data : bool, optional
     Whether to save the epoched data. Default is False.
    epochs_save_path : str or None, optional
     Path to save the epoched data. Required if save_data is True.
    epochs_data_fname : str or None, optional
     Filename to save the epoched data. Required if save_data is True.

    Returns
    -------
    epochs : instance of mne.Epochs
     The epoched MEG data.
    events : array, shape (n_events, 3)
     The events array.

    Raises
    ------
    ValueError
     If save_data is True and epochs_save_path or epochs_data_fname is not provided.
    """

    # Sanity check to save data
    if save_data and (not epochs_save_path or not epochs_data_fname):
        raise ValueError('Please provide path and filename to save data. If not, set save_data to false.')

    # Define events
    # if from_df:
    #     metadata, events, event_id, onset_times = define_events_from_df(subject=subject, meg_data=meg_data, epoch_id=epoch_id)
    # else:
    metadata, events, event_id, onset_times = define_events(subject=subject, meg_data=meg_data, epoch_id=epoch_id)

    # Reject based on channel amplitude
    if reject == False:
        # Setting reject parameter to False uses No rejection (None in mne will not reject)
        reject = None
    elif reject == None:
        # Default rejection parameter
        reject = dict(mag=subject.params.reject_amp)

    # Epoch data
    epochs = mne.Epochs(raw=meg_data, events=events, event_id=event_id, tmin=tmin, tmax=tmax, reject=None, proj=False,
                        event_repeated='drop', metadata=metadata, preload=True, baseline=baseline, reject_by_annotation=True)
    # Drop bad epochs
    epochs.drop_bad()

    if save_data:
        # Save epoched data
        epochs.reset_drop_log_selection()
        os.makedirs(epochs_save_path, exist_ok=True)
        epochs.save(epochs_save_path + epochs_data_fname, overwrite=True)

    return epochs, events, onset_times


def annotate_bad_intervals(meg_data, data_fname, data_type, sds, save_data=True):
    """
    Annotate bad intervals in the MEG data based on standard deviation thresholds.

    Parameters
    ----------
    meg_data : instance of mne.io.Raw
        The raw MEG data.
    data_fname : str
        Filename to save the annotated data.
    data_type : str
        Type of data ('ICA' or 'tsss').
    sds : float
        Standard deviation multiplier for thresholding.
    save_data : bool, optional
        Whether to save the annotated data. Default is True.

    Returns
    -------
    annot_data : instance of mne.io.Raw
        The annotated MEG data.
    bad_segments : list of lists
        List of bad segments with [start, duration].
    """
    # 0. Copy MEG data to avoid altering
    annot_data = meg_data.copy()

    # 1. Get the channel data as a NumPy array
    data = annot_data.copy().pick('meg').get_data()  # Shape: (n_channels, n_times)
    sfreq = annot_data.info['sfreq']  # Sampling frequency
    n_samples = data.shape[1]
    epoch_length = int(5 * sfreq)  # 5 seconds in samples

    # 2. Calculate the standard deviation for the entire signal
    std_per_channel_full = np.std(data, axis=1)  # Standard deviation across the entire signal

    # 3. Epoch the data into 5-second chunks
    bad_segments = []
    for start_idx in range(0, n_samples, epoch_length):
        end_idx = min(start_idx + epoch_length, n_samples)
        epoch_data = data[:, start_idx:end_idx]

        # Compute the standard deviation for the epoch
        std_per_channel_epoch = np.std(epoch_data, axis=1)

        # Check if any channel in the epoch exceeds the threshold
        if np.any(std_per_channel_epoch > std_per_channel_full * sds):
            start_time = start_idx / sfreq
            duration = (end_idx - start_idx) / sfreq
            bad_segments.append([start_time, duration])

    # 4. Create annotations for the "BAD" segments
    if bad_segments:
        onsets = [seg[0]  - annot_data.first_time for seg in bad_segments]  # Start times
        durations = [seg[1] for seg in bad_segments]  # Durations
        descriptions = ['BAD'] * len(bad_segments)  # "BAD" label

        annotations = mne.Annotations(onset=onsets,
                                      duration=durations,
                                      description=descriptions)

        # Get original annotations and substract first time
        orig_annotations = annot_data.annotations
        orig_annotations.onset = orig_annotations.onset - annot_data.first_time

        # 6. Add the annotations to the Raw object
        if annot_data.annotations:  # Check if there are existing annotations
            annot_data.set_annotations(orig_annotations + annotations)
        else:
            annot_data.set_annotations(annotations)

    # 5. (Optional) Save the data with annotations
    if save_data:
        print('Saving filtered data')
        if data_type == 'ICA':
            os.makedirs(paths.ica_annot_path, exist_ok=True)
            annot_data.save(paths.ica_annot_path + data_fname, overwrite=True)
        elif data_type == 'tsss':
            os.makedirs(paths.tsss_raw_annot_path, exist_ok=True)
            annot_data.save(paths.tsss_raw_annot_path + data_fname, overwrite=True)
        elif data_type == 'processed':
            os.makedirs(paths.processed_path_annot, exist_ok=True)
            annot_data.save(paths.processed_path_annot + data_fname, overwrite=True)

    return annot_data, bad_segments


def estimate_sources_cov(subject, meg_params, trial_params, filters, active_times, rank, bline_mode_subj, save_data, cov_save_path, cov_act_fname,
                         cov_baseline_fname, epochs_save_path, epochs_data_fname):

    try:
        # Load covariance matrix
        baseline_cov = mne.read_cov(fname=cov_save_path + cov_baseline_fname)
    except:
        # Load epochs
        try:
            epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
        except:
            # Load MEG
            meg_data = load.meg(subject_id=subject.subject_id, meg_params=meg_params)

            # Epoch data
            epochs, events, onset_times = epoch_data(subject=subject,
                                                     epoch_id=trial_params['epoch_id'],
                                                     meg_data=meg_data, tmin=trial_params['tmin'],
                                                     tmax=trial_params['tmax'],
                                                     baseline=trial_params['baseline'],
                                                     save_data=save_data,
                                                     epochs_save_path=epochs_save_path,
                                                     epochs_data_fname=epochs_data_fname,
                                                     reject=trial_params['reject'])

        # Compute covariance matrices
        baseline_cov = mne.cov.compute_covariance(epochs=epochs, tmin=trial_params['baseline'][0], tmax=trial_params['baseline'][1], method="shrunk", rank=dict(mag=rank))
        # Save
        if save_data:
            os.makedirs(cov_save_path, exist_ok=True)
            baseline_cov.save(fname=cov_save_path + cov_baseline_fname, overwrite=True)

    try:
        # Load covariance matrix
        active_cov = mne.read_cov(fname=cov_save_path + cov_act_fname)
    except:
        # Load epochs
        try:
            epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
        except:
            # Compute epochs
            # Load MEG
            meg_data = load.meg(subject_id=subject.subject_id, meg_params=meg_params)

            # Epoch data
            epochs, events, onset_times = epoch_data(subject=subject,
                                                     epoch_id=trial_params['epoch_id'],
                                                     meg_data=meg_data, tmin=trial_params['tmin'],
                                                     tmax=trial_params['tmax'],
                                                     baseline=trial_params['baseline'],
                                                     save_data=save_data,
                                                     epochs_save_path=epochs_save_path,
                                                     epochs_data_fname=epochs_data_fname,
                                                     reject=trial_params['reject'])

        # Compute covariance matrices
        active_cov = mne.cov.compute_covariance(epochs=epochs, tmin=active_times[0], tmax=active_times[1], method="shrunk", rank=dict(mag=rank))
        # Save
        if save_data:
            os.makedirs(cov_save_path, exist_ok=True)
            active_cov.save(fname=cov_save_path + cov_act_fname, overwrite=True)

    # Compute sources and apply baseline
    stc_base = mne.beamformer.apply_lcmv_cov(baseline_cov, filters)
    stc_act = mne.beamformer.apply_lcmv_cov(active_cov, filters)

    if bline_mode_subj == 'mean':
        stc = stc_act - stc_base
    elif bline_mode_subj == 'ratio':
        stc = stc_act / stc_base
    elif bline_mode_subj == 'db':
        stc = stc_act / stc_base
        stc.data = 10 * np.log10(stc.data)
    else:
        stc = stc_act

    return stc


def run_source_permutations_test(src, stc, source_data, subject, exp_info, save_regions, fig_path, surf_vol, p_threshold=0.05, n_permutations=1024, desired_tval='TFCE',
                                 mask_negatives=False):

    # Return variables
    stc_all_cluster_vis, significant_voxels, significance_mask, t_thresh_name, time_label = None, None, None, None, None

    # Compute source space adjacency matrix
    print("Computing adjacency matrix")
    adjacency_matrix = mne.spatial_src_adjacency(src)

    # Transpose source_fs_data from shape (subjects x space x time) to shape (subjects x time x space)
    source_data_default = source_data.swapaxes(1, 2)

    # Define the t-value threshold for cluster formation
    if desired_tval == 'TFCE':
        t_thresh = dict(start=0, step=0.1)
    else:
        df = len(exp_info.subjects_ids) - 1  # degrees of freedom for the test
        t_thresh = stats.distributions.t.ppf(1 - desired_tval / 2, df=df)

    # Run permutations
    T_obs, clusters, cluster_p_values, H0 = clu = spatio_temporal_cluster_1samp_test(X=source_data_default,
                                                                                     n_permutations=n_permutations,
                                                                                     adjacency=adjacency_matrix,
                                                                                     n_jobs=4, threshold=t_thresh)

    # Select the clusters that are statistically significant at p
    good_clusters_idx = np.where(cluster_p_values < p_threshold)[0]
    good_clusters = [clusters[idx] for idx in good_clusters_idx]
    significant_pvalues = [cluster_p_values[idx] for idx in good_clusters_idx]

    if len(good_clusters):

        # variable for figure fnames and p_values as title
        if type(t_thresh) == dict:
            time_label = f'{np.round(np.mean(significant_pvalues), 4)} +- {np.round(np.std(significant_pvalues), 4)}'
            t_thresh_name = 'TFCE'
        else:
            time_label = str(significant_pvalues)
            t_thresh_name = round(t_thresh, 2)

        # Get vertices from source space
        fsave_vertices = [s["vertno"] for s in src]

        # Select clusters for visualization
        stc_all_cluster_vis = summarize_clusters_stc(clu=clu, p_thresh=p_threshold, tstep=stc.tstep, vertices=fsave_vertices, subject=subject)

        # Get significant clusters
        significance_mask = np.where(stc_all_cluster_vis.data[:, 0] == 0)[0]
        significant_voxels = np.where(stc_all_cluster_vis.data[:, 0] != 0)[0]

        # Get significant AAL and brodmann regions from mni space
        if save_regions:
            os.makedirs(fig_path, exist_ok=True)
            significant_regions_df = get_regions_from_mni(src_default=src, significant_voxels=significant_voxels, save_path=fig_path, surf_vol=surf_vol,
                                                                            t_thresh_name=t_thresh_name, p_threshold=p_threshold, masked_negatves=mask_negatives)

    return stc_all_cluster_vis, significant_voxels, significance_mask, t_thresh_name, time_label, p_threshold


def get_regions_from_mni(src_default, significant_voxels, save_path, t_thresh_name, p_threshold, surf_vol, masked_negatves=False):

    # Get all source space used voxels locations (meters -> mm)
    if surf_vol == 'volume':
        used_voxels_mm = src_default[0]['rr'][src_default[0]['inuse'].astype(bool)] * 1000
    elif surf_vol == 'surface':
        used_voxels_mm = np.vstack((src_default[0]['rr'][src_default[0]['inuse'].astype(bool)] * 1000, src_default[1]['rr'][src_default[1]['inuse'].astype(bool)] * 1000))

    # Get significant voxels mni locations
    significant_voxels_mm = used_voxels_mm[significant_voxels]

    # Restrict to atlases range
    atlas_range = {0: (-270, 89), 1: (-341, 90), 2: (-251, 108)}

    # X coord
    significant_voxels_mm[significant_voxels_mm[:, 0] < atlas_range[0][0], 0] = atlas_range[0][0]
    significant_voxels_mm[significant_voxels_mm[:, 0] > atlas_range[0][1], 0] = atlas_range[0][1]
    # Y coord
    significant_voxels_mm[significant_voxels_mm[:, 1] < atlas_range[1][0], 1] = atlas_range[1][0]
    significant_voxels_mm[significant_voxels_mm[:, 1] > atlas_range[1][1], 1] = atlas_range[1][1]
    # Z coord
    significant_voxels_mm[significant_voxels_mm[:, 2] < atlas_range[2][0], 2] = atlas_range[2][0]
    significant_voxels_mm[significant_voxels_mm[:, 2] > atlas_range[2][1], 2] = atlas_range[2][1]

    atlas = AtlasBrowser("AAL3")
    significant_regions = atlas.find_regions(significant_voxels_mm)

    # Atlas regions
    all_aals = [region for region in significant_regions]
    aals_count = [(region, all_aals.count(region)) for region in significant_regions]
    sig_aals = list(set(aals_count))

    # Define dictonary to save regions
    save_dict = {'aal': [region[0] for region in sig_aals], 'aal occurence': [region[1] for region in sig_aals]}

    # Define save dataframe
    significant_regions_df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in save_dict.items()]))

    # Save
    if masked_negatves:
        fname = f'sig_t{t_thresh_name}_p{p_threshold}_{surf_vol}masked.csv'
    else:
        fname = f'sig_t{t_thresh_name}_p{p_threshold}_{surf_vol}.csv'
    significant_regions_df.to_csv(save_path + fname)

    return significant_regions_df


#---------- MTRF -----------#
def get_bad_annot_array(meg_data, subj_path, fname, save_var=True):
    # Get bad annotations times
    bad_annotations_idx = [i for i, annot in enumerate(meg_data.annotations.description) if
                           ('bad' in annot or 'BAD' in annot)]
    bad_annotations_time = meg_data.annotations.onset[bad_annotations_idx]
    bad_annotations_duration = meg_data.annotations.duration[bad_annotations_idx]
    bad_annotations_endtime = bad_annotations_time + bad_annotations_duration

    bad_indexes = []
    for i in range(len(bad_annotations_time)):
        bad_annotation_span_idx = np.where(
            np.logical_and((meg_data.times > bad_annotations_time[i]), (meg_data.times < bad_annotations_endtime[i])))[0]
        bad_indexes.append(bad_annotation_span_idx)

    # Flatten all indexes and convert to array
    bad_indexes = functions_general.flatten_list(bad_indexes)
    bad_indexes = np.array(bad_indexes)

    # Make bad annotations binary array
    bad_annotations_array = np.ones(len(meg_data.times))
    if len(bad_indexes):
        bad_annotations_array[bad_indexes] = 0

    # Save arrays
    if save_var:
        save.var(var=bad_annotations_array, path=subj_path, fname=fname)

    return bad_annotations_array


def make_mtrf_input(input_arrays, var_name, subject, meg_data, bad_annotations_array,
                    subj_path, fname, save_var=True):

    secondary_variable = False
    if '-' in var_name:
        secondary_variable = True
        var_name_2 = var_name.split('-')[1]
        var_name = var_name.split('-')[0]

    # if var_name in ['steering', 'gas', 'brake', 'steering_std', 'gas_std', 'brake_std']:
    if 'steering' in var_name or 'gas' in var_name or 'brake' in var_name:

        feature_name = var_name.replace('_std', '').replace('_der', '')
        meg_params = {'data_type': 'processed'}
        raw = load.meg(subject_id=subject.subject_id, meg_params=meg_params)
        input_array = raw.get_data(picks=feature_name)[0, :]

        if '_der' in var_name:
            # Compute derivative
            input_array = np.gradient(input_array)

        if '_std' in var_name:
            # Standarize variable
            input_array = (input_array - np.mean(input_array)) / np.std(input_array)

    else:
        # Define events
        # if from_df:
        #     metadata, events, event_id, onset_times = define_events_from_df(subject=subject, meg_data=meg_data, epoch_id=var_name)
        # else:
        metadata, events, event_id, onset_times = define_events(subject=subject, meg_data=meg_data, epoch_id=var_name)

        # Make input arrays as 0
        input_array = np.zeros(len(meg_data.times))
        # Get events samples index
        evt_idxs = events[:, 0] - int(meg_data.first_time * meg_data.info['sfreq'])
        # Set those indexes as 1 or as variable
        if secondary_variable:
            if var_name_2 == 'on_mirror':
                input_array[evt_idxs] = metadata[var_name_2].astype(int)
            if var_name_2 == 'stimulus_present':
                input_array[evt_idxs] = metadata[var_name_2].astype(int)
            if '_X_' in var_name_2:
                var_name_21 = var_name_2.split('_X_')[0]
                var_name_22 = var_name_2.split('_X_')[1]
                input_array = input_arrays[f'{var_name}-{var_name_21}'] * input_arrays[f'{var_name}-{var_name_22}']
            else:
                input_array[evt_idxs] = metadata[var_name_2].to_numpy()
            var_name = f'{var_name}-{var_name_2}'  # Overwrite varname to add in corresponding dictionary key
        else:
            input_array[evt_idxs] = 1

    # Exclude bad annotations
    input_array = input_array * bad_annotations_array
    # Save to all input arrays dictionary
    input_arrays[var_name] = input_array

    # Save arrays
    if save_var:
        save.var(var=input_array, path=subj_path, fname=fname)

    return input_arrays


def fit_mtrf(meg_data, tmin, tmax, model_input, chs_id, standarize=True, fit_power=False, alpha=0, n_jobs=4):

    # Define mTRF model
    rf = ReceptiveField(tmin, tmax, meg_data.info['sfreq'], estimator=alpha, scoring='corrcoef', n_jobs=n_jobs)

    if chs_id != 'misc':
        # Get subset channels data as array
        picks = functions_general.pick_chs(chs_id=chs_id, info=meg_data.info)
        meg_sub = meg_data.copy().pick(picks)
        # Apply hilbert and extract envelope
        if fit_power:
            meg_sub.load_data()
            meg_sub = meg_sub.apply_hilbert(envelope=True)
    else:
        meg_sub = meg_data.copy()
        # Apply hilbert and extract envelope
        if fit_power:
            meg_sub = meg_sub.apply_hilbert(envelope=True, picks='VE')

    meg_data_array = meg_sub.get_data()

    if standarize:
        # Standarize data
        print('Computing z-score...')
        meg_data_array = np.expand_dims(meg_data_array, axis=0)  # Need shape (n_epochs, n_channels n_times)
        meg_data_array = mne.decoding.Scaler(info=meg_sub.info, scalings='mean').fit_transform(meg_data_array)
        meg_data_array = meg_data_array.squeeze()
    # Transpose to input the model
    meg_data_array = meg_data_array.T

    # Fit TRF
    rf.fit(model_input, meg_data_array)

    return rf


def load_input_array_feature(feature, meg_params, subj_path, use_saved_data=True):
    fname_var = (f"{feature}_array.pkl")

    if meg_params.get('downsample'):
        fname_var = fname_var.replace('_array.pkl', f'_{meg_params.get("downsample")}_array.pkl')

    if os.path.exists(subj_path + fname_var) and use_saved_data:
        feature_array = load.var(file_path=subj_path + fname_var)
        print(f"Loaded input array for {feature} from \n"
              f"{fname_var}")
        return feature_array, fname_var

    else:
        return False, fname_var

def compute_trf(subject, meg_data, trf_params, meg_params, features, alpha=None, all_chs_regions=['frontal', 'temporal', 'parietal', 'occipital'],
                use_saved_data=True, save_data=False, trf_path=None, trf_fname=None):

    if not alpha:
        alpha = trf_params['alpha']

    print(f"Computing TRF for {trf_params['input_features']}")

    # Bad annotations filepath
    subj_path = paths.save_path + f'TRF/{subject.subject_id}/'
    fname_bad_annot = f'bad_annot_array.pkl'

    if os.path.exists(subj_path + fname_bad_annot) and use_saved_data:
        bad_annotations_array = load.var(subj_path + fname_bad_annot)
        print(f'Loaded bad annotations array')
    else:
        print(f'Computing bad annotations array...')
        bad_annotations_array = get_bad_annot_array(meg_data=meg_data, subj_path=subj_path, fname=fname_bad_annot)

    input_arrays = {}
    subj_path = paths.save_path + f'TRF/{subject.subject_id}/'
    for feature in features:
        feature_data, fname_var = load_input_array_feature(feature=feature, meg_params=meg_params, subj_path=subj_path, use_saved_data=use_saved_data)
        if isinstance(feature_data, np.ndarray):
            input_arrays[feature] = feature_data
        else:
            print(f'Computing input array for {feature}...')
            input_arrays = make_mtrf_input(input_arrays=input_arrays, var_name=feature,
                                           subject=subject, meg_data=meg_data,
                                           bad_annotations_array=bad_annotations_array,
                                           subj_path=subj_path, fname=fname_var)

    # Concatenate input arrays as one
    model_input = np.array([input_arrays[key] for key in input_arrays.keys()]).T
    # All regions or selected (multiple) regions
    if meg_params['chs_id'] == 'mag' or '_' in meg_params['chs_id']:
        # rf as a dictionary containing the rf of each region
        rf = {}
        # iterate over regions
        for chs_subset in all_chs_regions:
            # Use only regions in channels id, or all in case of chs_id == 'mag'
            print(f'Fitting mTRF for region {chs_subset}')
            rf[chs_subset] = fit_mtrf(meg_data=meg_data, tmin=trf_params['tmin'], tmax=trf_params['tmax'], alpha=alpha, fit_power=trf_params['fit_power'],
                                                         model_input=model_input, chs_id=chs_subset, standarize=trf_params['standarize'], n_jobs=4)
    # One region
    else:
        rf = fit_mtrf(meg_data=meg_data, tmin=trf_params['tmin'], tmax=trf_params['tmax'], alpha=alpha, fit_power=trf_params['fit_power'],
                                                             model_input=model_input, chs_id=meg_params['chs_id'], standarize=trf_params['standarize'], n_jobs=4)
    # Save TRF
    if save_data:
        save.var(var=rf, path=trf_path, fname=trf_fname)

    return rf


def make_trf_evoked(subject, rf, meg_data, trf_params, meg_params, feature_index, feature, feature_evokeds=None, display_figs=False, plot_individuals=True, save_fig=True, fig_path=None):

    # All or multiple regions
    if meg_params['chs_id'] == 'mag' or '_' in meg_params['chs_id']:

        # Define evoked from TRF list to concatenate all
        subj_evoked_list = []

        # iterate over regions
        for chs_idx, chs_subset in enumerate(rf.keys()):
            # Get channels subset info
            picks = functions_general.pick_chs(chs_id=chs_subset, info=meg_data.info)
            meg_sub = meg_data.copy().pick(picks)

            # Get TRF coeficients from chs subset
            trf = rf[chs_subset].coef_[:, feature_index, :]

            if chs_idx == 0:
                # Define evoked object from arrays of TRF
                subj_evoked = mne.EvokedArray(data=trf, info=meg_sub.info, tmin=trf_params['tmin'], baseline=trf_params['baseline'])
            else:
                # Append evoked object from arrays of TRF to list, to concatenate all
                subj_evoked_list.append(mne.EvokedArray(data=trf, info=meg_sub.info, tmin=trf_params['tmin'], baseline=trf_params['baseline']))

        # Concatenate evoked from al regions
        subj_evoked = subj_evoked.add_channels(subj_evoked_list)

    elif meg_params['chs_id'] == 'misc':
        trf = np.expand_dims(rf.coef_[feature_index, :], axis=0)
        # Define evoked objects from arrays of TRF
        subj_evoked = mne.EvokedArray(data=trf, info=meg_data.info, tmin=trf_params['tmin'], baseline=trf_params['baseline'])
    else:
        trf = rf.coef_[:, feature_index, :]
        # Define evoked objects from arrays of TRF
        subj_evoked = mne.EvokedArray(data=trf, info=meg_data.info, tmin=trf_params['tmin'], baseline=trf_params['baseline'])

    # Append for Grand average
    if feature_evokeds != None:
        feature_evokeds[feature].append(subj_evoked)

    # Plot
    if plot_individuals:
        fig = subj_evoked.plot(spatial_colors=True, gfp=True, show=display_figs, xlim=(trf_params['tmin'], trf_params['tmax']), titles=feature)
        fig.suptitle(f'{feature}')

        if save_fig:
            # Save
            fig_path_subj = fig_path + f'{subject.subject_id}/'
            fname = f"{feature}_{meg_params['chs_id']}"
            save.fig(fig=fig, fname=fname, path=fig_path_subj)

    return feature_evokeds


def parse_trf_to_evoked(subject, rf, meg_data, trf_params, meg_params, sub_idx, feature_evokeds=None, display_figs=False, plot_individuals=False, save_fig=False, fig_path=None):
    """
    Get model coeficients as separate responses to each feature.

    Parameters
    ----------
    subject
    rf
    meg_data
    evokeds
    trf_params
    trial_params
    meg_params
    display_figs
    plot_individuals
    save_fig
    fig_path

    Returns
    -------

    """

    # Sanity check
    if save_fig and not fig_path:
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    elements = feature_evokeds.keys()
    feature_index = 0
    for feature in elements:

        feature_evokeds = make_trf_evoked(subject=subject, rf=rf, meg_data=meg_data, feature_evokeds=feature_evokeds,
                                  trf_params=trf_params, feature_index=feature_index, feature=feature, meg_params=meg_params,
                                  plot_individuals=plot_individuals, display_figs=display_figs, save_fig=save_fig, fig_path=fig_path)
        feature_index += 1

    if trf_params.get('add_features'):
        if sub_idx == 0:
            feature_evokeds['+'.join(trf_params['add_features'])] = []
        data = np.zeros_like(feature_evokeds[trf_params['add_features'][0]][sub_idx].data)
        for i in range(len(trf_params['add_features'])):
            data += feature_evokeds[trf_params['add_features'][i]][sub_idx].data

        # Define evoked objects from arrays of TRF
        info = feature_evokeds[trf_params['add_features'][0]][sub_idx].info
        add_evoked = mne.EvokedArray(data=data, info=info, tmin=trf_params['tmin'], baseline=trf_params['baseline'])

        feature_evokeds['+'.join(trf_params['add_features'])].append(add_evoked)

    return feature_evokeds


def trf_grand_average(feature_evokeds, trf_params, meg_params, display_figs=False, save_fig=True, fig_path=None):

    # Sanity check
    if save_fig and not fig_path:
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    grand_avg = {}
    for feature in feature_evokeds.keys():
        # Compute grand average
        grand_avg[feature] = mne.grand_average(feature_evokeds[feature], interpolate_bads=False)

        # Plot
        fig = grand_avg[feature].plot(spatial_colors=True, gfp=True, show=display_figs, xlim=(trf_params['tmin'], trf_params['tmax']), titles=feature)

        if save_fig:
            # Save
            fname = f"{feature}_GA_{meg_params['chs_id']}"
            save.fig(fig=fig, fname=fname, path=fig_path)

    return grand_avg
