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
from sklearn.model_selection import KFold
from scipy import stats as stats
import scipy.signal
from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc, permutation_cluster_1samp_test
from mni_to_atlas import AtlasBrowser


exp_info = setup.exp_info()


def get_experiment_phase_mask(subject_id, meg_data):
    """Load experiment phase times and create phase masks for the MEG recording.

    Phases:
    - CF: Car Following only (CF active, DA inactive, Audio inactive)
    - DA: Divided Attention (DA active; Audio takes priority if overlapping)
    - Audio: Audiobook (Audio active)

    Parameters
    ----------
    subject_id : str
        Participant ID (e.g., '17359').
    meg_data : mne.io.Raw
        MEG data to determine time axis.

    Returns
    -------
    dict : {'CF': np.ndarray, 'DA': np.ndarray, 'Audio': np.ndarray}
        Boolean masks for each phase, same length as meg_data.times.
    """
    import csv

    sfreq = meg_data.info['sfreq']
    n_times = len(meg_data.times)
    times = meg_data.times + meg_data.first_time  # absolute times in seconds

    # Load CF times
    with open(paths.bh_path + 'CF_EVENT_TIME_1.csv', 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    cf_ids = rows[0]
    cf_onsets = [float(x) for x in rows[1]]
    cf_offsets = [float(x) for x in rows[2]]
    cf_idx = cf_ids.index(str(subject_id)) if str(subject_id) in cf_ids else None

    # Load Audio times
    with open(paths.bh_path + 'AUDIO_EVENT_TIME_1.csv', 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    audio_ids = rows[0]
    audio_onsets = [float(x) for x in rows[1]]
    audio_offsets = [float(x) for x in rows[2]]
    audio_idx = audio_ids.index(str(subject_id)) if str(subject_id) in audio_ids else None

    # Load DA times (60 stimuli, DA phase = first stimulus to last + 4s)
    with open(paths.bh_path + 'DA_EVENT_TIME_1.csv', 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    da_ids = rows[0]
    da_idx = da_ids.index(str(subject_id)) if str(subject_id) in da_ids else None

    # Build boolean masks
    cf_mask = np.zeros(n_times, dtype=bool)
    da_mask = np.zeros(n_times, dtype=bool)
    audio_mask = np.zeros(n_times, dtype=bool)

    if cf_idx is not None:
        cf_on, cf_off = cf_onsets[cf_idx], cf_offsets[cf_idx]
        cf_mask = (times >= cf_on) & (times <= cf_off)
    else:
        import warnings
        warnings.warn(f'Subject {subject_id} not found in CF_EVENT_TIME_1.csv. CF mask will be all zeros.')

    if da_idx is not None:
        da_stim_times = [float(rows[r][da_idx]) for r in range(1, len(rows)) if rows[r][da_idx].strip()]
        da_on = min(da_stim_times)
        da_off = max(da_stim_times) + 4.0  # last stimulus + 4s duration
        da_mask = (times >= da_on) & (times <= da_off)
    else:
        import warnings
        warnings.warn(f'Subject {subject_id} not found in DA_EVENT_TIME_1.csv. DA mask will be all zeros.')

    if audio_idx is not None:
        audio_on, audio_off = audio_onsets[audio_idx], audio_offsets[audio_idx]
        audio_mask = (times >= audio_on) & (times <= audio_off)
    else:
        import warnings
        warnings.warn(f'Subject {subject_id} not found in AUDIO_EVENT_TIME_1.csv. Audio mask will be all zeros.')

    # Audio takes priority over DA when they overlap
    da_mask = da_mask & ~audio_mask

    # CF-only: CF active but neither DA nor Audio
    cf_only_mask = cf_mask & ~da_mask & ~audio_mask

    print(f'  Phase masks - CF: {cf_only_mask.sum()/sfreq:.1f}s, '
          f'DA: {da_mask.sum()/sfreq:.1f}s, Audio: {audio_mask.sum()/sfreq:.1f}s')

    return {'CF': cf_only_mask, 'DA': da_mask, 'Audio': audio_mask}


def expand_features(input_features):
    """Expand input_features dict into a flat list of feature names.

    Handles:
    - None value: just the feature name (e.g., 'audio_env_std': None -> ['audio_env_std'])
    - List of phase tags: base feature + phase-tagged variants
      (e.g., 'fix': ['CF', 'DA', 'Audio'] -> ['fix', 'fix_CF', 'fix_DA', 'fix_Audio'])
    - List of secondary variables (strings with no phase match):
      (e.g., 'fix': ['on_mirror'] -> ['fix', 'fix-on_mirror'])

    Returns
    -------
    list : Flat list of all feature names.
    """
    phase_names = {'CF', 'DA', 'Audio'}
    features = []
    for feature, values in input_features.items():
        features.append(feature)
        if isinstance(values, (list, tuple)):
            for val in values:
                if val in phase_names:
                    features.append(f'{feature}_{val}')
                else:
                    features.append(f'{feature}-{val}')
        elif isinstance(values, str):
            if values in phase_names:
                features.append(f'{feature}_{values}')
            else:
                features.append(f'{feature}-{values}')
    return features


# ---------- Per-feature TRF duration helpers ---------- #
def get_feature_tmin_tmax(feature, trf_params):
    """Resolve per-feature tmin/tmax from trf_params.

    trf_params['tmin'] and trf_params['tmax'] can be:
    - scalar: used for all features
    - dict: {'default': val, 'feature_name': val, ...}
      Features not in dict use the 'default' key value.
    """
    tmin = trf_params['tmin']
    tmax = trf_params['tmax']
    if isinstance(tmin, dict):
        feat_tmin = tmin.get(feature, tmin.get('default'))
    else:
        feat_tmin = tmin
    if isinstance(tmax, dict):
        feat_tmax = tmax.get(feature, tmax.get('default'))
    else:
        feat_tmax = tmax
    return feat_tmin, feat_tmax


def get_feature_initial_time(feature, initial_time):
    """Resolve per-feature initial_time for brain plots.

    initial_time can be:
    - scalar or None: used for all features
    - dict: {'default': val, 'feature_name': val, ...}
      Features not in dict use the 'default' key (None as fallback).
    Values can be a single number/None or a list of numbers to generate
    multiple brain plots at different time points.

    Returns
    -------
    list : Always returns a list of time values (even for a single value).
    """
    if isinstance(initial_time, dict):
        val = initial_time.get(feature, initial_time.get('default', None))
    else:
        val = initial_time
    # Normalize to list
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val]


def group_features_by_duration(features, trf_params):
    """Group features by their (tmin, tmax) pair for fitting separate models."""
    from collections import OrderedDict
    groups = OrderedDict()
    for feature in features:
        tmin, tmax = get_feature_tmin_tmax(feature, trf_params)
        key = (tmin, tmax)
        if key not in groups:
            groups[key] = []
        groups[key].append(feature)
    return groups


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
            # Filter by preceding saccade class if specified
            if 'short' in epoch_id:
                metadata = metadata[metadata['prev_sac_class'] == 'short'].reset_index(drop=True)
            elif 'long' in epoch_id:
                metadata = metadata[metadata['prev_sac_class'] == 'long'].reset_index(drop=True)

        if 'sac' in epoch_id:
            # Load df of events
            metadata = subject.saccades()
            # Filter by saccade class if specified
            if 'short' in epoch_id:
                metadata = metadata[metadata['sac_class'] == 'short'].reset_index(drop=True)
            elif 'long' in epoch_id:
                metadata = metadata[metadata['sac_class'] == 'long'].reset_index(drop=True)

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
                # Left button from button/left/green channel (rising edge detection)
                meg_params_full = {'data_type': 'processed'}
                raw_full = load.meg(subject_id=subject.subject_id, meg_params=meg_params_full)
                button_data = raw_full.get_data(picks='button/left/green')[0, :]
                crossings = np.where(np.diff((button_data > 0.5).astype(int)) == 1)[0]
                onset_times = crossings / raw_full.info['sfreq']

                onset_description = ['left_but'] * len(onset_times)
                task_duration = [0] * len(onset_times)

                stim_annotations = mne.Annotations(onset=onset_times,
                                                   duration=task_duration,
                                                   description=onset_description)

                meg_data_copy = meg_data.copy()
                meg_data_copy.set_annotations(stim_annotations)

                events, event_id = mne.events_from_annotations(meg_data_copy, verbose=False)
                epoch_keys = ['left_but']

            elif 'right_but' == epoch_id:
                # Right button from button/right/yellow channel (rising edge detection)
                meg_params_full = {'data_type': 'processed'}
                raw_full = load.meg(subject_id=subject.subject_id, meg_params=meg_params_full)
                button_data = raw_full.get_data(picks='button/right/yellow')[0, :]
                crossings = np.where(np.diff((button_data > 0.5).astype(int)) == 1)[0]
                onset_times = crossings / raw_full.info['sfreq']

                onset_description = ['right_but'] * len(onset_times)
                task_duration = [0] * len(onset_times)

                stim_annotations = mne.Annotations(onset=onset_times,
                                                   duration=task_duration,
                                                   description=onset_description)

                meg_data_copy = meg_data.copy()
                meg_data_copy.set_annotations(stim_annotations)

                events, event_id = mne.events_from_annotations(meg_data_copy, verbose=False)
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


def epoch_data(subject, epoch_id, meg_data, tmin, tmax, baseline=(0, 0), reject=None, save_data=False, epochs_save_path=None, epochs_data_fname=None):
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
    epochs = mne.Epochs(raw=meg_data.copy().pick('mag'), events=events, event_id=event_id, tmin=tmin, tmax=tmax, reject=None, proj=False,
                        event_repeated='drop', metadata=metadata, preload=True, baseline=baseline, reject_by_annotation=True)
    # Drop bad epochs
    # epochs.drop_bad()

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
        t_thresh = dict(start=0, step=0.2)
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

    # Check for phase suffix (_CF, _DA, _Audio)
    phase_suffix = None
    phase_names = ['_CF', '_DA', '_Audio']
    for ps in phase_names:
        if var_name.endswith(ps):
            phase_suffix = ps[1:]  # Remove leading underscore
            base_feature = var_name[:-len(ps)]
            break

    if phase_suffix is not None:
        # Phase-tagged feature: multiply base feature by phase mask
        # First ensure the base feature array exists
        if base_feature not in input_arrays:
            base_fname = fname.replace(var_name, base_feature)
            input_arrays = make_mtrf_input(
                input_arrays=input_arrays, var_name=base_feature,
                subject=subject, meg_data=meg_data,
                bad_annotations_array=bad_annotations_array,
                subj_path=subj_path, fname=base_fname, save_var=save_var)

        # Get phase masks (cached on meg_data to avoid re-reading files)
        if not hasattr(meg_data, '_phase_masks'):
            meg_data._phase_masks = get_experiment_phase_mask(subject.subject_id, meg_data)

        phase_mask = meg_data._phase_masks[phase_suffix]
        input_array = input_arrays[base_feature] * phase_mask.astype(float)

        # Exclude bad annotations
        input_array = input_array * bad_annotations_array
        input_arrays[var_name] = input_array

        if save_var:
            save.var(var=input_array, path=subj_path, fname=fname)

        return input_arrays

    secondary_variable = False
    if '-' in var_name:
        secondary_variable = True
        var_name_2 = var_name.split('-')[1]
        var_name = var_name.split('-')[0]

    # if var_name in ['steering', 'gas', 'brake', 'steering_std', 'gas_std', 'brake_std']:
    if 'Steering' in var_name or 'Gas' in var_name or 'Brake' in var_name:

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

    elif 'audio_env' in var_name:
        # Audio envelope from synchronized video audio (preprocess_audio.py)
        meg_params_full = {'data_type': 'processed'}
        raw = load.meg(subject_id=subject.subject_id, meg_params=meg_params_full)

        if 'AudioEnvVideo' in raw.ch_names:
            input_array = raw.get_data(picks='AudioEnvVideo')[0, :]
        else:
            # Fallback: compute on-the-fly from video
            import preprocess_audio
            raw, _ = preprocess_audio.extract_and_sync_audio(
                subject.subject_id, raw, save_fig=False, plot=False)
            input_array = raw.get_data(picks='AudioEnvVideo')[0, :]

        # Mask engine noise: only when _msk suffix is used
        if '_msk' in var_name:
            sfreq = raw.info['sfreq']
            win = int(5 * sfreq)
            rolling_power = np.convolve(input_array**2, np.ones(win)/win, mode='same')
            rolling_rms = np.sqrt(rolling_power)
            threshold = np.median(rolling_rms) * 1.5
            audio_active = rolling_rms > threshold

            diff = np.diff(audio_active.astype(int))
            starts = np.where(diff == 1)[0] / sfreq
            stops = np.where(diff == -1)[0] / sfreq
            if audio_active[0]:
                starts = np.concatenate([[0], starts])
            if audio_active[-1]:
                stops = np.concatenate([stops, [len(input_array) / sfreq]])
            intervals = [(s, e) for s, e in zip(starts, stops) if (e - s) > 10]

            times_arr = np.arange(len(input_array)) / sfreq
            mask = np.zeros(len(input_array), dtype=bool)
            for s, e in intervals:
                mask |= (times_arr >= s) & (times_arr <= e)
            input_array[~mask] = 0
            input_array[input_array < 0.02] = 0

        if '_der' in var_name:
            input_array = np.gradient(input_array)

        if '_std' in var_name:
            input_array = (input_array - np.mean(input_array)) / np.std(input_array)
        elif '_norm' in var_name:
            input_array = (input_array - np.min(input_array)) / (np.max(input_array) - np.min(input_array))

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


def extract_best_alpha(rf):
    """Extract best alpha from a fitted ReceptiveField.

    Returns
    -------
    float or None
        The best alpha value, or None if alpha was fixed (not cross-validated).
    """
    if hasattr(rf, 'best_alpha_'):
        return rf.best_alpha_
    return None


def save_alpha_report(rf_results, subject_id, save_path, is_multi_region=False):
    """Save a text report of the selected alpha values per duration group.

    Parameters
    ----------
    rf_results : list of dict
        Each dict has 'rf', 'features', 'tmin', 'tmax', and optionally 'best_alpha'.
    subject_id : str
        Subject identifier (or 'GA' for grand average summary).
    save_path : str
        Directory to save the report.
    is_multi_region : bool
        If True, rf is a dict of region -> RF objects.
    """
    lines = [f'Alpha Report - {subject_id}', '=' * 50, '']
    for group in rf_results:
        features_str = ', '.join(group['features'])
        lines.append(f'Features: {features_str}')
        lines.append(f'  Duration: tmin={group["tmin"]}, tmax={group["tmax"]}')
        if is_multi_region and isinstance(group.get('best_alpha'), dict):
            for region, alpha_val in group['best_alpha'].items():
                lines.append(f'  Best alpha ({region}): {alpha_val}')
        else:
            lines.append(f'  Best alpha: {group.get("best_alpha", "N/A")}')
        lines.append('')

    os.makedirs(save_path, exist_ok=True)
    report_path = os.path.join(save_path, f'alpha_report_{subject_id}.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Alpha report saved to {report_path}')


def fit_mtrf(meg_data, tmin, tmax, model_input, chs_id, standarize=True, fit_power=False, alpha=0,
             n_splits=5, cv_aggregate='mean_fisher', n_jobs=4):
    """Fit an mTRF/encoding model, cross-validating the ridge parameter (alpha).

    Parameters
    ----------
    alpha : float | array-like
        If a single float, used directly as the ridge estimator. If a list/array of
        candidate values, the best alpha is selected by k-fold cross-validation.
    n_splits : int
        Number of cross-validation folds used to select alpha (contiguous temporal
        blocks, no shuffling, to respect the time-series structure). Default 5.
    cv_aggregate : {'mean_fisher', 'mean', 'pool'}
        How the per-fold validation correlations are aggregated into a single score
        per alpha:
        - 'mean_fisher' : Fisher z-transform each fold's mean correlation, average,
          then inverse-transform (recommended; robust to non-stationarity, reduces
          the bias of averaging correlations). Default.
        - 'mean'        : plain average of the per-fold mean correlations.
        - 'pool'        : concatenate the held-out predictions/targets across folds
          and compute a single correlation over the whole dataset.
    """

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
            meg_sub = meg_sub.apply_hilbert(envelope=True, picks='misc')

    meg_data_array = meg_sub.get_data()

    if standarize:
        # Standarize data
        print('Computing z-score...')
        meg_data_array = np.expand_dims(meg_data_array, axis=0)  # Need shape (n_epochs, n_channels n_times)
        meg_data_array = mne.decoding.Scaler(info=meg_sub.info, scalings='mean').fit_transform(meg_data_array)
        meg_data_array = meg_data_array.squeeze()
    # Transpose to input the model
    meg_data_array = meg_data_array.T

    sfreq = meg_data.info['sfreq']

    # Cross-validate alpha if a list of candidates is provided
    if isinstance(alpha, (list, tuple, np.ndarray)):
        alphas = np.array(alpha, dtype=float)
        n_times = meg_data_array.shape[0]

        # Contiguous temporal folds (no shuffle) to respect the time-series structure
        n_folds = int(min(n_splits, n_times))
        n_folds = max(n_folds, 2)
        kf = KFold(n_splits=n_folds, shuffle=False)

        def _columnwise_corr(a, b):
            """Pearson correlation per column (channel/target) of (n_times, n_chs)."""
            a = a - a.mean(axis=0, keepdims=True)
            b = b - b.mean(axis=0, keepdims=True)
            num = np.sum(a * b, axis=0)
            den = np.sqrt(np.sum(a ** 2, axis=0) * np.sum(b ** 2, axis=0))
            with np.errstate(invalid='ignore', divide='ignore'):
                r = num / den
            return r

        # Evaluate all alpha candidates by k-fold CV
        scores = []
        for a in alphas:
            fold_scores = []          # used by 'mean' / 'mean_fisher'
            pooled_pred, pooled_true = [], []   # used by 'pool'
            for train_idx, test_idx in kf.split(model_input):
                rf_cv = ReceptiveField(tmin, tmax, sfreq, estimator=float(a),
                                       scoring='corrcoef', n_jobs=n_jobs)
                rf_cv.fit(model_input[train_idx], meg_data_array[train_idx])

                if cv_aggregate == 'pool':
                    # Collect valid (edge-trimmed) predictions/targets to pool later
                    y_pred = rf_cv.predict(model_input[test_idx])
                    mask = rf_cv.valid_samples_
                    pooled_pred.append(np.atleast_2d(y_pred[mask]))
                    pooled_true.append(np.atleast_2d(meg_data_array[test_idx][mask]))
                else:
                    # nanmean across channels: a target with no variance in this fold
                    # yields a NaN corrcoef; ignore it instead of poisoning the mean.
                    fold_score = np.nanmean(rf_cv.score(model_input[test_idx],
                                                        meg_data_array[test_idx]))
                    fold_scores.append(fold_score)
                del rf_cv

            if cv_aggregate == 'pool':
                pred = np.concatenate(pooled_pred, axis=0)
                true = np.concatenate(pooled_true, axis=0)
                score = np.nanmean(_columnwise_corr(true, pred))
            elif cv_aggregate == 'mean_fisher':
                fs = np.array(fold_scores, dtype=float)
                fs = fs[np.isfinite(fs)]
                if fs.size == 0:
                    score = np.nan
                else:
                    # Fisher z-average (clip to avoid +/-inf at |r|==1)
                    z = np.arctanh(np.clip(fs, -0.999999, 0.999999))
                    score = float(np.tanh(np.mean(z)))
            else:  # 'mean'
                score = float(np.nanmean(fold_scores))

            scores.append(score)
            print(f'    alpha={a}: score={score:.6f}')

        # Pick largest alpha within 1% of the best score
        scores = np.array(scores, dtype=float)
        finite = np.isfinite(scores)
        if not np.any(finite):
            # All scores are NaN (e.g. degenerate folds): fall back to the median
            # candidate alpha instead of crashing.
            best_alpha = float(np.median(alphas))
            print(f'  WARNING: all CV scores are NaN; falling back to median alpha {best_alpha}')
        else:
            best_score = np.nanmax(scores)
            tolerance = 0.01 * abs(best_score)
            # Among alphas with a finite score >= best_score - tolerance, pick the largest
            within_tol = alphas[finite & (scores >= best_score - tolerance)]
            best_alpha = float(np.max(within_tol))
            print(f'  Best alpha: {best_alpha} '
                  f'(best_score={best_score:.6f}, tol={tolerance:.6f}, '
                  f'{n_folds}-fold {cv_aggregate})')

        estimator = best_alpha
    else:
        estimator = alpha

    # Final fit on all data
    rf = ReceptiveField(tmin, tmax, sfreq, estimator=estimator, scoring='corrcoef', n_jobs=n_jobs)
    rf.fit(model_input, meg_data_array)
    # Store selected alpha for later retrieval
    rf.best_alpha_ = estimator if isinstance(alpha, (list, tuple, np.ndarray)) else None

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

    # Cross-validation settings for alpha selection (with sensible defaults)
    n_splits = trf_params.get('cv_n_splits', 5)
    cv_aggregate = trf_params.get('cv_aggregate', 'mean_fisher')

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

    # Group features by (tmin, tmax) duration and fit separate models per group
    duration_groups = group_features_by_duration(features, trf_params)
    is_multi_region = meg_params['chs_id'] == 'mag' or '_' in meg_params['chs_id']

    rf_results = []
    for (group_tmin, group_tmax), group_features in duration_groups.items():
        group_input = np.array([input_arrays[f] for f in group_features]).T

        # All regions or selected (multiple) regions
        if is_multi_region:
            group_rf = {}
            best_alpha = {}
            for chs_subset in all_chs_regions:
                print(f'Fitting mTRF for region {chs_subset} (tmin={group_tmin}, tmax={group_tmax})')
                group_rf[chs_subset] = fit_mtrf(meg_data=meg_data, tmin=group_tmin, tmax=group_tmax, alpha=alpha,
                                                fit_power=trf_params['fit_power'], model_input=group_input,
                                                chs_id=chs_subset, standarize=trf_params['standarize'],
                                                n_splits=n_splits, cv_aggregate=cv_aggregate, n_jobs=4)
                region_alpha = extract_best_alpha(group_rf[chs_subset])
                if region_alpha is not None:
                    best_alpha[chs_subset] = region_alpha
                    print(f'  Best alpha ({chs_subset}): {region_alpha}')
        # One region
        else:
            group_rf = fit_mtrf(meg_data=meg_data, tmin=group_tmin, tmax=group_tmax, alpha=alpha,
                                fit_power=trf_params['fit_power'], model_input=group_input,
                                chs_id=meg_params['chs_id'], standarize=trf_params['standarize'],
                                n_splits=n_splits, cv_aggregate=cv_aggregate, n_jobs=4)
            best_alpha = extract_best_alpha(group_rf)
            if best_alpha is not None:
                print(f'  Best alpha: {best_alpha}')

        rf_results.append({
            'rf': group_rf,
            'features': group_features,
            'tmin': group_tmin,
            'tmax': group_tmax,
            'best_alpha': best_alpha,
        })

    # Save alpha report
    if trf_path:
        save_alpha_report(rf_results, subject.subject_id, trf_path, is_multi_region=is_multi_region)

    # Save TRF
    if save_data:
        save.var(var=rf_results, path=trf_path, fname=trf_fname)

    return rf_results


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
        margin = trf_params.get('plot_margin', 0)
        fig = subj_evoked.plot(spatial_colors=True, gfp=True, show=display_figs, xlim=(trf_params['tmin'] + margin, trf_params['tmax'] - margin), titles=feature)
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
    Supports both legacy rf (single ReceptiveField / dict of region->RF)
    and new per-duration-group format (list of group dicts from compute_trf).
    """

    # Sanity check
    if save_fig and not fig_path:
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    elements = feature_evokeds.keys()

    if isinstance(rf, list):
        # Per-duration-group structure from compute_trf
        feature_map = {}
        for group in rf:
            for idx, feat in enumerate(group['features']):
                feature_map[feat] = {
                    'rf': group['rf'],
                    'feature_index': idx,
                    'tmin': group['tmin'],
                    'tmax': group['tmax'],
                }

        for feature in elements:
            if feature in feature_map:
                fmap = feature_map[feature]
                feat_trf_params = dict(trf_params)
                feat_trf_params['tmin'] = fmap['tmin']
                feat_trf_params['tmax'] = fmap['tmax']
                feat_trf_params['baseline'] = (fmap['tmin'], fmap['tmax'])
                feature_evokeds = make_trf_evoked(subject=subject, rf=fmap['rf'], meg_data=meg_data,
                                                  feature_evokeds=feature_evokeds,
                                                  trf_params=feat_trf_params, feature_index=fmap['feature_index'],
                                                  feature=feature, meg_params=meg_params,
                                                  plot_individuals=plot_individuals, display_figs=display_figs,
                                                  save_fig=save_fig, fig_path=fig_path)
    else:
        # Legacy format (single RF or dict of region->RF)
        feature_index = 0
        for feature in elements:
            feat_tmin, feat_tmax = get_feature_tmin_tmax(feature, trf_params)
            feat_trf_params = dict(trf_params)
            feat_trf_params['tmin'] = feat_tmin
            feat_trf_params['tmax'] = feat_tmax
            feat_trf_params['baseline'] = (feat_tmin, feat_tmax)
            feature_evokeds = make_trf_evoked(subject=subject, rf=rf, meg_data=meg_data, feature_evokeds=feature_evokeds,
                                      trf_params=feat_trf_params, feature_index=feature_index, feature=feature, meg_params=meg_params,
                                      plot_individuals=plot_individuals, display_figs=display_figs, save_fig=save_fig, fig_path=fig_path)
            feature_index += 1

    if trf_params.get('add_features'):
        if sub_idx == 0:
            feature_evokeds['+'.join(trf_params['add_features'])] = []
        data = np.zeros_like(feature_evokeds[trf_params['add_features'][0]][sub_idx].data)
        for i in range(len(trf_params['add_features'])):
            data += feature_evokeds[trf_params['add_features'][i]][sub_idx].data

        # Use the first add_feature's duration for the combined evoked
        info = feature_evokeds[trf_params['add_features'][0]][sub_idx].info
        first_feat = trf_params['add_features'][0]
        first_tmin, first_tmax = get_feature_tmin_tmax(first_feat, trf_params)
        add_evoked = mne.EvokedArray(data=data, info=info, tmin=first_tmin, baseline=(first_tmin, first_tmax))

        feature_evokeds['+'.join(trf_params['add_features'])].append(add_evoked)

    return feature_evokeds


def trf_grand_average(feature_evokeds, trf_params, meg_params, display_figs=False, save_fig=True, fig_path=None):

    # Sanity check
    if save_fig and not fig_path:
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    grand_avg = {}
    for feature in feature_evokeds.keys():
        # Compute grand average
        grand_avg[feature] = mne.grand_average(feature_evokeds[feature], interpolate_bads=True)

        # Use per-feature time limits and plot margin
        feat_tmin, feat_tmax = get_feature_tmin_tmax(feature, trf_params)
        margin = trf_params.get('plot_margin', 0)
        t0, t1 = feat_tmin + margin, feat_tmax - margin

        # Plot
        fig = grand_avg[feature].plot(spatial_colors=True, gfp=True, show=display_figs,
                                      xlim=(t0, t1), titles=feature)

        # Rescale y-axis to fit only the visible (trimmed) time window
        ax = fig.get_axes()[0]
        time_mask = (grand_avg[feature].times >= t0) & (grand_avg[feature].times <= t1)
        visible_data = grand_avg[feature].data[:, time_mask]
        data_full_max = np.abs(grand_avg[feature].data).max()
        current_ylim = ax.get_ylim()
        if data_full_max > 0:
            scaling = max(abs(current_ylim[0]), abs(current_ylim[1])) / data_full_max
        else:
            scaling = 1.0
        data_max = np.abs(visible_data).max()
        if data_max > 0:
            pad = data_max * scaling * 0.1
            ax.set_ylim(-data_max * scaling - pad, data_max * scaling + pad)

        if save_fig:
            # Save
            fname = f"{feature}_GA_{meg_params['chs_id']}"
            save.fig(fig=fig, fname=fname, path=fig_path)

    return grand_avg


def get_parcellation_adjacency(ch_names, surf_vol, subject='fsaverage', subjects_dir=None,
                               parc='aparc.a2009s', label_positions=None,
                               vol_n_neighbors=6, vol_dist_factor=1.5):
    """Build a region-level adjacency matrix for cluster-based permutation stats
    on parcellated (label) source data.

    The cluster-permutation framework needs to know which regions are neighbors
    so that spatially contiguous effects can be grouped into clusters.

    - Surface parcellations ('parcellation'): adjacency is *anatomical*. Two
      labels are neighbors if they share a border on the cortical surface, i.e.
      an edge of the surface triangulation connects a vertex of one label to a
      vertex of the other (computed per hemisphere; hemispheres are not linked).
    - Volume parcellations ('vol_parcellation') or when surface info is missing:
      adjacency is *geometric*. Regions are connected to nearby regions based on
      the Euclidean distance between their centroids (a distance threshold plus a
      k-nearest-neighbours floor so no region is left isolated).

    Parameters
    ----------
    ch_names : list of str
        Region/label names in the SAME order as the data channels.
    surf_vol : str
        'parcellation' (surface) or 'vol_parcellation' (volume).
    subject : str
        FreeSurfer subject used for surface adjacency (e.g. 'fsaverage').
    subjects_dir : str
        FreeSurfer subjects directory.
    parc : str
        Surface parcellation name (e.g. 'aparc', 'aparc.a2009s').
    label_positions : dict or None
        {region_name: np.array([x, y, z])} centroid positions. Required for
        volume/geometric adjacency.
    vol_n_neighbors : int
        Minimum number of nearest neighbours each region connects to (volume).
    vol_dist_factor : float
        Distance threshold as a multiple of the median nearest-neighbour
        distance (volume).

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix
        (n_regions, n_regions) boolean adjacency including self-connections,
        aligned to `ch_names`.
    """
    import scipy.sparse
    from scipy.spatial import cKDTree

    n = len(ch_names)
    name_to_idx = {name: i for i, name in enumerate(ch_names)}
    adjacency = scipy.sparse.lil_matrix((n, n), dtype=bool)
    for i in range(n):
        adjacency[i, i] = True

    if surf_vol == 'parcellation':
        if subjects_dir is None:
            raise ValueError('subjects_dir is required for surface parcellation adjacency.')

        print(f'Computing anatomical label adjacency ({parc}) on {subject}...')
        labels = mne.read_labels_from_annot(subject, parc=parc, subjects_dir=subjects_dir, verbose=False)

        for hemi in ('lh', 'rh'):
            surf_file = os.path.join(subjects_dir, subject, 'surf', f'{hemi}.white')
            rr, tris = mne.read_surface(surf_file)

            # Map each surface vertex to a label index (within this hemisphere)
            hemi_labels = [lab for lab in labels if lab.hemi == hemi]
            vlabel = np.full(rr.shape[0], -1, dtype=int)
            for li, label in enumerate(hemi_labels):
                vlabel[label.vertices] = li

            # All triangle edges -> label pairs that differ -> bordering labels
            edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [0, 2]]], axis=0)
            la = vlabel[edges[:, 0]]
            lb = vlabel[edges[:, 1]]
            valid = (la >= 0) & (lb >= 0) & (la != lb)
            border_pairs = np.unique(np.sort(np.stack([la[valid], lb[valid]], axis=1), axis=1), axis=0)

            for a_id, b_id in border_pairs:
                na, nb = hemi_labels[a_id].name, hemi_labels[b_id].name
                if na in name_to_idx and nb in name_to_idx:
                    ia, ib = name_to_idx[na], name_to_idx[nb]
                    adjacency[ia, ib] = True
                    adjacency[ib, ia] = True

    else:
        # Geometric adjacency from centroid positions
        if label_positions is None:
            raise ValueError('label_positions is required for volume/geometric adjacency.')
        missing = [name for name in ch_names if name not in label_positions]
        if missing:
            raise ValueError(f'label_positions is missing {len(missing)} region(s), '
                             f'e.g. {missing[:3]}')

        print(f'Computing geometric centroid adjacency for {n} regions...')
        pos = np.array([label_positions[name] for name in ch_names])
        tree = cKDTree(pos)

        # Distance-threshold adjacency
        nn_dist = tree.query(pos, k=min(2, n))[0]
        nn_dist = nn_dist[:, -1] if nn_dist.ndim == 2 else nn_dist
        thresh = np.median(nn_dist) * vol_dist_factor
        for a, b in tree.query_pairs(r=thresh):
            adjacency[a, b] = True
            adjacency[b, a] = True

        # k-nearest-neighbour floor so no region is isolated
        k = min(vol_n_neighbors + 1, n)
        knn_idx = tree.query(pos, k=k)[1]
        knn_idx = np.atleast_2d(knn_idx)
        for i in range(n):
            for j in knn_idx[i, 1:]:
                adjacency[i, int(j)] = True
                adjacency[int(j), i] = True

    adjacency = adjacency.tocsr()
    n_edges = int((adjacency.sum() - n) / 2)
    print(f'Region adjacency: {n} regions, {n_edges} neighbour links '
          f'(mean degree {2 * n_edges / n:.1f})')
    return adjacency


def run_permutations_test(data, pval_threshold, t_thresh, adj_matrix=None, n_permutations=1024, seed=42):

    # Clusters out type
    if type(t_thresh) == dict:
        out_type = 'indices'
    else:
        out_type = 'mask'

    significant_pvalues = None

    # Permutations cluster test (TFCE if t_thresh as dict)
    t_tfce, clusters, p_tfce, H0 = permutation_cluster_1samp_test(X=data, threshold=t_thresh, n_permutations=n_permutations, adjacency=adj_matrix,
                                                                  out_type=out_type, seed=seed, n_jobs=4)

    # Make clusters mask
    if type(t_thresh) == dict:
        # If TFCE use p-vaues of voxels directly
        p_tfce = p_tfce.reshape(data.shape[-2:])  # Reshape to data's shape
        clusters_mask = p_tfce < pval_threshold

    else:
        # Get significant clusters
        good_clusters_idx = np.where(p_tfce < pval_threshold)[0]
        significant_clusters = [clusters[idx] for idx in good_clusters_idx]
        significant_pvalues = [p_tfce[idx] for idx in good_clusters_idx]

        # Reshape to data's shape by adding all clusters into one bool array
        clusters_mask = np.zeros(data[0].shape)
        if len(significant_clusters):
            for significant_cluster in significant_clusters:
                clusters_mask += significant_cluster
        clusters_mask = clusters_mask.astype(bool)

    return clusters_mask, significant_pvalues


def evoked_to_parcellation_stc(evoked, parc, subject_code, subjects_dir, spacing):
    """Convert label-named Evoked to a full-surface STC with parcellation coloring.

    Each vertex within a label gets the label's TRF coefficient value,
    producing uniform coloring per region in brain plots.
    """
    # Read the full surface source space
    fname_src = paths.sources_path + subject_code + f'/{subject_code}_surface_{spacing}-src.fif'
    src_full = mne.read_source_spaces(fname_src)

    labels = mne.read_labels_from_annot(subject_code, parc=parc, subjects_dir=subjects_dir)

    # Map Evoked channel names to data
    ch_data = {ch: evoked.data[i] for i, ch in enumerate(evoked.ch_names)}

    n_times = evoked.data.shape[1]
    lh_verts = src_full[0]['vertno']
    rh_verts = src_full[1]['vertno']

    lh_data = np.zeros((len(lh_verts), n_times))
    rh_data = np.zeros((len(rh_verts), n_times))

    for label in labels:
        if label.name not in ch_data:
            continue
        if label.hemi == 'lh':
            mask = np.isin(lh_verts, label.vertices)
            lh_data[mask] = ch_data[label.name]
        else:
            mask = np.isin(rh_verts, label.vertices)
            rh_data[mask] = ch_data[label.name]

    data = np.vstack([lh_data, rh_data])
    stc = mne.SourceEstimate(data, vertices=[lh_verts, rh_verts],
                             tmin=evoked.times[0],
                             tstep=1 / evoked.info['sfreq'])
    return stc, src_full