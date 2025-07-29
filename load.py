import functions_analysis
import paths
import setup
import os
import pathlib
import pickle
import mne
import functions_general
import glob
import plot_general



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


def preproc_meg_data(subject_id, task, preload=False):
    """
    Preprocessed MEG data for parent subject as raw instance of MNE.
    """

    # Subject preprocessed data path
    file_path = pathlib.Path(os.path.join(paths.preproc_path, f'{task}_{subject_id}_bad_ch_removed_meg.fif'))

    # Try to load preprocessed data
    try:
        print('\nLoading Preprocessed MEG data')
        meg_data = mne.io.read_raw_fif(file_path, preload=preload)
    except:
        raise ValueError(f'No previous preprocessed data found for subject {subject_id} in {file_path}')

    return meg_data


def filtered_data(subject_id, band_id, task, method='iir', data_type='ICA', preload=True, save_data=True):

    if data_type == 'ICA':
        filtered_path = paths.filtered_path_ica + f'{band_id}/'
    elif data_type == 'ICA_annot':
        filtered_path = paths.filtered_path_ica_annot + f'{band_id}/'
    elif data_type == 'tsss':
        filtered_path = paths.filtered_path_tsss + f'{band_id}/'
    elif data_type == 'tsss_annot':
        filtered_path = paths.filtered_path_tsss_annot + f'{band_id}/'
    elif data_type == 'RAW':
        filtered_path = paths.filtered_path_raw + f'{band_id}/'
    else:
        raise ValueError(f'Invalid data_type. Should be either ICA, ICA_annot tsss, tsss_annot or RAW. Got {data_type} instead')

    filtered_meg_data_fname = f'Subject_{subject_id}_method_{method}_meg.fif'

    # Try to load filtered data
    try:
        print(f'Loading filtered data in band {band_id} for subject {subject_id}')
        # Load data
        filtered_data = mne.io.read_raw_fif(filtered_path + filtered_meg_data_fname, preload=preload)

    except:
        print(f'No previous filtered {data_type} data found for subject {subject_id} in band {band_id}.\n'
              f'Filtering data...')

        meg_data = meg_type(subject_id=subject_id, task=task, data_type=data_type, preload=preload)

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

    # ICA data path
    data_path = paths.ica_path + f'{task}_{subject_id}_raw_ica_hfc_meg.fif'

    # Try to load ica data
    try:
        print(f'Loading ICA data for subject {subject_id}')
        # Load data
        meg_data = mne.io.read_raw_fif(data_path, preload=preload)
    except:
        raise ValueError(f'No previous data found for subject {subject_id}')

    return meg_data


def ica_annot_data(subject_id, task, sds=3, preload=True, save_data=True, plot_bad_segments=False):

    # ICA data path
    data_fname =  f'{task}_{subject_id}_raw_ica_hfc_meg_annot_{sds}.fif'

    # Try to load ica data
    try:
        print(f'Loading data for subject {subject_id}')
        # Load data
        annot_data = mne.io.read_raw_fif(paths.ica_annot_path + data_fname, preload=preload)
    except:
        print(f'No previous ICA annotated data found for subject {subject_id} in {paths.ica_annot_path + data_fname}')
        print(f'Running ICA and annotating data with default parameters {sds} SD...')

        # Load ICA data
        meg_data = ica_data(subject_id=subject_id, task=task, preload=True)

        # Annotate bad segments as per default parameters
        annot_data, bad_segments = functions_analysis.annotate_bad_intervals(meg_data, data_fname=data_fname, data_type='ICA', sds=sds, save_data=save_data)

        # Plot bad segments
        if plot_bad_segments:
            fig = plot_general.bad_segments(meg_data=annot_data, bad_segments=bad_segments, sds=sds)

    return annot_data


def tsss_raw_data(subject_id, task, preload=True):

    # ICA data path
    data_path = paths.tsss_raw_path + f'{subject_id}_{task}_raw_tsss_meg.fif'

    # Try to load ica data
    try:
        print(f'Loading TSSS data for subject {subject_id}')
        # Load data
        meg_data = mne.io.read_raw_fif(data_path, preload=preload)
    except:
        raise ValueError(f'No previous data found for subject {subject_id} in {data_path}')

    return meg_data


def tsss_raw_annot_data(subject_id, task, sds=3, preload=True, save_data=True, plot_bad_segments=False):

    # ICA data path
    data_fname =  f'{subject_id}_{task}_raw_tsss_meg_annot_{sds}.fif'

    # Try to load ica data
    try:
        print(f'Loading data for subject {subject_id}')
        # Load data
        annot_data = mne.io.read_raw_fif(paths.tsss_raw_annot_path + data_fname, preload=preload)
    except:
        print(f'No previous tsss annotated data found for subject {subject_id} in {paths.tsss_raw_annot_path + data_fname}')
        print(f'Running tsss and annotating data with default parameters {sds} SD...')

        # Load TSSS data
        meg_data = tsss_raw_data(subject_id=subject_id, task=task, preload=True)

        # Annotate bad segments as per default parameters
        annot_data, bad_segments = functions_analysis.annotate_bad_intervals(meg_data, data_fname=data_fname, data_type='tsss', sds=sds, save_data=save_data)

        # Plot bad segments
        if plot_bad_segments:
            fig = plot_general.bad_segments(meg_data=annot_data, bad_segments=bad_segments, sds=sds)

    return annot_data


def meg(subject_id, meg_params, task='DA', preload=True, save_data=True):

    # Define parameters from meg_params
    band_id = meg_params.get('band_id', None)
    data_type = meg_params.get('data_type', None)
    filter_sensors = meg_params.get('filter_sensors', None)
    filter_method = meg_params.get('filter_method', None)

    if band_id and filter_sensors:
        meg_data = filtered_data(subject_id=subject_id, task=task, data_type=data_type, band_id=band_id,
                                 save_data=save_data, method=filter_method)
    else:
        meg_data = meg_type(subject_id=subject_id, task=task, data_type=data_type, preload=preload)

    return meg_data


def meg_type(subject_id, task, data_type='ICA', preload=True):
    """
    Load MEG data for a given subject, task and data type.
    :param subject_id:
    :param task:
    :param data_type:
    :param preload:
    :return:
    """
    if data_type == 'ICA':
        meg_data = ica_data(subject_id=subject_id, task=task, preload=preload)
    elif data_type == 'ICA_annot':
        meg_data = ica_annot_data(subject_id=subject_id, task=task, preload=preload)
    elif data_type == 'tsss':
        meg_data = tsss_raw_data(subject_id=subject_id, task=task, preload=preload)
    elif data_type == 'tsss_annot':
        meg_data = tsss_raw_annot_data(subject_id=subject_id, task=task, preload=preload)
    elif data_type == 'RAW':
        meg_data = preproc_meg_data(subject_id=subject_id, task=task, preload=preload)
    else:
        raise ValueError(
            f'Invalid data_type. Should be either ICA, ICA_annot, tsss_raw, tsss_raw_annot or RAW. Got {data_type} instead')
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