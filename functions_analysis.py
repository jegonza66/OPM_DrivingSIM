import mne
import numpy as np
import os
import setup
from setup import exp_info

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

    if epoch_keys is None:
        # Define epoch keys as all events
        epoch_keys = []

        if 'CF' == epoch_id:
            # Get task onset times
            drive_onset_time = meg_data.annotations.onset[np.where(meg_data.annotations.description == 'drive')[0]][0]
            onset_times = [subject.exp_times['cf_start'] + drive_onset_time]
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
            onset_times = [subject.da_times['DA times'][0] + drive_onset_time]

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
            onset_times = [drive_onset_time]

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

            # Get epoch keys from epoch_id
            epoch_keys = epoch_id.split('+')
    else:
        # Get events from annotations
        events, event_id = mne.events_from_annotations(meg_data, verbose=False)

    # Get events and ids matching selection
    metadata, events, events_id = mne.epochs.make_metadata(events=events, event_id=event_id, row_events=epoch_keys, tmin=0, tmax=0, sfreq=meg_data.info['sfreq'])

    return metadata, events, events_id


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
    metadata, events, events_id = define_events(subject=subject, meg_data=meg_data, epoch_id=epoch_id)

    # Reject based on channel amplitude
    if reject == False:
        # Setting reject parameter to False uses No rejection (None in mne will not reject)
        reject = None
    elif reject == None:
        # Default rejection parameter
        reject = dict(mag=subject.params.reject_amp)

    # Epoch data
    epochs = mne.Epochs(raw=meg_data, events=events, event_id=events_id, tmin=tmin, tmax=tmax, reject=reject,
                        event_repeated='drop', metadata=metadata, preload=True, baseline=baseline)
    # Drop bad epochs
    # epochs.drop_bad()

    if save_data:
        # Save epoched data
        epochs.reset_drop_log_selection()
        os.makedirs(epochs_save_path, exist_ok=True)
        epochs.save(epochs_save_path + epochs_data_fname, overwrite=True)

    return epochs, events