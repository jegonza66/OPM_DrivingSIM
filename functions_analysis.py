import mne
import numpy as np
import os
import setup
from setup import exp_info
import paths
import load
import save
import functions_general
from mne.decoding import ReceptiveField

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

        elif 'DA' == epoch_id:
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
            events[:, 0] = events[:, 0] - functions_general.find_nearest(meg_data.times, meg_data.first_time)[0]  # Adjust event times to start from 0

            # Get epoch keys from epoch_id
            epoch_keys = epoch_id.split('+')
    else:
        # Get events from annotations

        events, event_id = mne.events_from_annotations(meg_data, verbose=False)

    # Get events and ids matching selection
    metadata, events, events_id = mne.epochs.make_metadata(events=events, event_id=event_id, row_events=epoch_keys, tmin=0, tmax=0, sfreq=meg_data.info['sfreq'])

    return metadata, events, events_id, onset_times


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
    metadata, events, events_id, onset_times = define_events(subject=subject, meg_data=meg_data, epoch_id=epoch_id)

    # Reject based on channel amplitude
    if reject == False:
        # Setting reject parameter to False uses No rejection (None in mne will not reject)
        reject = None
    elif reject == None:
        # Default rejection parameter
        reject = dict(mag=subject.params.reject_amp)

    # Epoch data
    epochs = mne.Epochs(raw=meg_data, events=events, event_id=events_id, tmin=tmin, tmax=tmax, reject=None,
                        event_repeated='drop', metadata=metadata, preload=True, baseline=baseline, reject_by_annotation=False)
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
    annot_data.pick('meg')

    # 1. Get the channel data as a NumPy array
    data = annot_data.get_data()  # Shape: (n_channels, n_times)
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
            np.logical_and((meg_data.times > bad_annotations_time[i]), (meg_data.times < bad_annotations_endtime[i])))[
            0]
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

    # Define events
    metadata, events, _, _ = define_events(subject=subject, epoch_id=var_name, meg_data=meg_data)
    # Make input arrays as 0
    input_array = np.zeros(len(meg_data.times))
    # Get events samples index
    evt_idxs = events[:, 0]
    # Set those indexes as 1
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

    # Get subset channels data as array
    picks = functions_general.pick_chs(chs_id=chs_id, info=meg_data.info)
    meg_sub = meg_data.copy().pick(picks)

    # Apply hilbert and extract envelope
    if fit_power:
        meg_sub = meg_sub.apply_hilbert(envelope=True)

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


def compute_trf(subject, meg_data, trf_params, meg_params, all_chs_regions=['frontal', 'temporal', 'parietal', 'occipital'],
                save_data=False, trf_path=None, trf_fname=None):

    print(f"Computing TRF for {trf_params['input_features']}")

    # Bad annotations filepath
    subj_path = paths.save_path + f'TRF/{subject.subject_id}/'
    fname_bad_annot = f'bad_annot_array.pkl'

    try:
        bad_annotations_array = load.var(subj_path + fname_bad_annot)
        print(f'Loaded bad annotations array')
    except:
        print(f'Computing bad annotations array...')
        bad_annotations_array = get_bad_annot_array(meg_data=meg_data, subj_path=subj_path, fname=fname_bad_annot)

    # Iterate over input features
    input_arrays = {}
    for feature in trf_params['input_features']:

        subj_path = paths.save_path + f'TRF/{subject.subject_id}/'
        fname_var = f"{feature}_array.pkl"

        try:
            input_arrays[feature] = load.var(file_path=subj_path + fname_var)
            print(f"Loaded input array for {feature}")

        except:
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
            if chs_subset in meg_params['chs_id'] or meg_params['chs_id'] == 'mag':
                print(f'Fitting mTRF for region {chs_subset}')
                rf[chs_subset] = fit_mtrf(meg_data=meg_data, tmin=trf_params['tmin'], tmax=trf_params['tmax'], alpha=trf_params['alpha'], fit_power=trf_params['fit_power'],
                                                             model_input=model_input, chs_id=chs_subset, standarize=trf_params['standarize'], n_jobs=4)
    # One region
    else:
        rf = fit_mtrf(meg_data=meg_data, tmin=trf_params['tmin'], tmax=trf_params['tmax'], alpha=trf_params['alpha'], fit_power=trf_params['fit_power'],
                                                             model_input=model_input, chs_id=meg_params['chs_id'], standarize=trf_params['standarize'], n_jobs=4)
    # Save TRF
    if save_data:
        save.var(var=rf, path=trf_path, fname=trf_fname)

    return rf


def make_trf_evoked(subject, rf, meg_data, trf_params, meg_params, evokeds=None, display_figs=False, plot_individuals=True, save_fig=True, fig_path=None):
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

    # Variables to store coefficients, and "evoked" data of the subject.
    trf = {}
    subj_evoked = {}
    subj_evoked_list = {}
    for i, feature in enumerate(trf_params['input_features']):

        # All or multiple regions
        if meg_params['chs_id'] == 'mag' or '_' in meg_params['chs_id']:

            # Define evoked from TRF list to concatenate all
            subj_evoked_list[feature] = []

            # iterate over regions
            for chs_idx, chs_subset in enumerate(rf.keys()):
                # Get channels subset info
                picks = functions_general.pick_chs(chs_id=chs_subset, info=meg_data.info)
                meg_sub = meg_data.copy().pick(picks)

                # Get TRF coeficients from chs subset
                trf[feature] = rf[chs_subset].coef_[:, i, :]

                if chs_idx == 0:
                    # Define evoked object from arrays of TRF
                    subj_evoked[feature] = mne.EvokedArray(data=trf[feature], info=meg_sub.info, tmin=trf_params['tmin'], baseline=trf_params['baseline'])
                else:
                    # Append evoked object from arrays of TRF to list, to concatenate all
                    subj_evoked_list[feature].append(mne.EvokedArray(data=trf[feature], info=meg_sub.info, tmin=trf_params['tmin'], baseline=trf_params['baseline']))

            # Concatenate evoked from al regions
            subj_evoked[feature] = subj_evoked[feature].add_channels(subj_evoked_list[feature])

        else:
            trf[feature] = rf.coef_[:, i, :]
            # Define evoked objects from arrays of TRF
            subj_evoked[feature] = mne.EvokedArray(data=trf[feature], info=meg_sub.info, tmin=trf_params['tmin'], baseline=trf_params['baseline'])

        # Append for Grand average
        if evokeds != None:
            evokeds[feature].append(subj_evoked[feature])

        # Plot
        if plot_individuals:
            fig = subj_evoked[feature].plot(spatial_colors=True, gfp=True, show=display_figs, xlim=(trf_params['tmin'], trf_params['tmax']), titles=feature)
            fig.suptitle(f'{feature}')

        if save_fig:
            # Save
            fig_path_subj = fig_path + f'{subject.subject_id}/'
            fname = f"{feature}_{meg_params['chs_id']}"
            save.fig(fig=fig, fname=fname, path=fig_path_subj)

    return subj_evoked, evokeds


def trf_grand_average(feature_evokeds, trf_params, trial_params, meg_params, display_figs=False, save_fig=True, fig_path=None):

    # Sanity check
    if save_fig and not fig_path:
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    grand_avg = {}
    for feature in trf_params['input_features']:
        # Compute grand average
        grand_avg[feature] = mne.grand_average(feature_evokeds[feature], interpolate_bads=True)
        plot_times_idx = np.where((grand_avg[feature].times > trf_params['tmin']) & (grand_avg[feature].times < trf_params['tmax']))[0]
        data = grand_avg[feature].get_data()[:, plot_times_idx]

        # Plot
        fig = grand_avg[feature].plot(spatial_colors=True, gfp=True, show=display_figs, xlim=(trf_params['tmin'], trf_params['tmax']), titles=feature)

        if save_fig:
            # Save
            fname = f"{feature}_GA_{meg_params['chs_id']}"
            save.fig(fig=fig, fname=fname, path=fig_path)

    return grand_avg
