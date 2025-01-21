import paths
import setup
import os
import pathlib
import pickle
import mne
import functions_general
import glob


def config(path, fname):
    """
    Try and load the run configuration and setup information.
    If no previous configuration file was saved, setup config obj.

    Parameters
    ----------
    path: str
        The path to the directory where configuration file is stored.
    fname: str
        The filename for the configuration file.

    Returns
    -------
    config: class
        Class containgn the run configuration and setup information.
    """

    try:
        # Load
        filepath = path + fname
        f = open(filepath, 'rb')
        config = pickle.load(f)
        f.close()

        # Set save config as false
        config.update_config = False

    except:
        # Create if no previous configuration file
        config = setup.config()

    return config


def var(file_path):
    """
    Load variable from specified path

    Parameters
    ----------
    file_path: str
        The path to the file to load.

    Returns
    -------
    var: any
        The loaded variable.
    """
    # Load
    f = open(file_path, 'rb')
    var = pickle.load(f)
    f.close()

    return var


# Raw MEG data
def load_raw_meg_data(subject_id, task, preload=True):
    """
    MEG data for parent subject as Raw instance of MNE.
    """

    print('\nLoading Raw MEG data')
    # Get subject path
    subj_data_path = pathlib.Path(os.path.join(paths.opm_path, subject_id, f'{subject_id}_{task}_meg.fif'))
    if os.path.exists(subj_data_path):
        raw = mne.io.read_raw_fif(subj_data_path, preload=preload)
    # Missing data
    else:
        raise ValueError('No fif files found in subject directory: {}'.format(subj_data_path))

    return raw


def preproc_meg_data(subject_id, preload=False):
    """
    Preprocessed MEG data for parent subject as raw instance of MNE.
    """

    # Subject preprocessed data path
    file_path = pathlib.Path(os.path.join(paths.preproc_path, subject_id, f'Subject_{subject_id}_meg.fif'))

    # Try to load preprocessed data
    try:
        print('\nLoading Preprocessed MEG data')
        meg_data = mne.io.read_raw_fif(file_path, preload=preload)
    except:
        raise ValueError(f'No previous preprocessed data found for subject {subject_id}')

    return meg_data


def filtered_data(subject_id, band_id, task, method='iir', use_ica_data=True, preload=True, save_data=True):

    if use_ica_data:
        filtered_path = paths.filtered_path_ica + f'{band_id}/{subject_id}/'
    else:
        filtered_path = paths.filtered_path_raw + f'{band_id}/{subject_id}/'

    filtered_meg_data_fname = f'Subject_{subject_id}_method_{method}_meg.fif'

    # Try to load filtered data
    try:
        print(f'Loading filtered data in band {band_id} for subject {subject_id}')
        # Load data
        filtered_data = mne.io.read_raw_fif(filtered_path + filtered_meg_data_fname, preload=preload)

    except:
        print(f'No previous filtered data found for subject {subject_id} in band {band_id}.\n'
              f'Filtering data...')

        if use_ica_data:
            meg_data = ica_data(subject_id=subject_id, task=task, preload=True)
        else:
            meg_data = preproc_meg_data(preload=True)

        l_freq, h_freq = functions_general.get_freq_band(band_id)
        if method:
            filtered_data = meg_data.filter(l_freq=l_freq, h_freq=h_freq, method=method)
        else:
            filtered_data = meg_data.filter(l_freq=l_freq, h_freq=h_freq)

        if save_data:
            print('Saving filtered data')
            # Save MEG
            os.makedirs(filtered_path, exist_ok=True)
            filtered_data.save(filtered_path + filtered_meg_data_fname, overwrite=True)

    return filtered_data


def ica_data(subject_id, task, preload=True):

    path = paths.ica_path + f'{subject_id}/'

    # Try to load ica data
    try:
        print(f'Loading ica data for subject {subject_id}')
        # Load data
        ica_data = mne.io.read_raw_fif(path + f'{task}_3_raw_ica_hfc_meg.fif', preload=preload)

    except:
        raise ValueError(f'No previous ica data found for subject {subject_id}')

    return ica_data


def meg(subject_id, task, data_type='ICA', band_id=None, filter_sensors=True, filter_method='iir', save_data=True):

    if data_type == 'ICA':
        if band_id and filter_sensors:
            meg_data = filtered_data(subject_id=subject_id, task=task, band_id=band_id, save_data=save_data,
                                     method=filter_method)
        else:
            meg_data = ica_data(subject_id=subject_id, task=task, preload=True)
    elif data_type == 'RAW':
        if band_id and filter_sensors:
            meg_data = filtered_data(subject_id=subject_id, task=task, band_id=band_id, use_ica_data=False,
                                     save_data=save_data, method=filter_method)
        else:
            meg_data = subject_id.load_preproc_meg_data()

    return meg_data


def time_frequency_range(file_path, l_freq, h_freq):

    # MS difference
    matching_files_ms = glob.glob(file_path)
    if len(matching_files_ms):
        for file in matching_files_ms:
            l_freq_file = int(file.split('_')[-3])
            h_freq_file = int(file.split('_')[-2])

            # If file contains desired frequencies, Load
            if l_freq_file <= l_freq and h_freq_file >= h_freq:
                time_frequency = mne.time_frequency.read_tfrs(file)[0]

                # Crop to desired frequencies
                time_frequency = time_frequency.crop(fmin=l_freq, fmax=h_freq)
                break
            else:
                raise ValueError('No file found with desired frequency range')
    else:
        raise ValueError('No file found with desired frequency range')

    return time_frequency