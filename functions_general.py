import numpy as np
from scipy.signal import butter, lfilter
import mne


def find_nearest(array, values):
    """
    Find the nearest element in an array to a given value

    Parameters
    ----------
    array: ndarray
        The 1D array to look in for the nearest value.
    values: int, float, list or 1D array
        If int or float, use that value to find neares elemen in array and return index and element as int and array.dtype
        If list or array, iterate over values and return arrays of indexes and elements nearest to each value.

    Returns
    -------
    idx: int
      The index of the element in the array that is nearest to the given value.
    element: int float
        The nearest element to the specified value
    """
    array = np.asarray(array)

    if isinstance(values, float) or isinstance(values, int):
        idx = (np.abs(array - values)).argmin()
        return idx, array[idx]

    elif len(values):
        idxs = []
        elements = []
        for value in values:
            idx = (np.abs(array - value)).argmin()
            idxs.append(idx)
            elements.append(array[idx])
        return np.asarray(idxs), np.asarray(elements)


def find_previous(array, value):
    """
    Find the nearest element in an array to a given value

    Parameters
    ----------
    array: ndarray
        The 1D array to look in for the nearest value.

    Returns
    -------
    idx: int
      The index of the element in the array that is nearest to the given value.
    """

    array = np.asarray(array)
    idx = np.max(np.where(array - value <= 0)[0])
    return idx, array[idx]



def find_first_within(array, low_bound, up_bound):
    """
    Find the first element from an array in a certain interval

    Parameters
    ----------
    array: ndarray
        The 1D array to look in for the nearest value.
    low_bound: float
        the lower boundary of the search interval
    up_bound: float
        the upper boundary of the search interval

    Returns
    -------
    idx: int
      The index of the element in the array that is nearest to the given value.
    value: float
      The value of the array in the found index.
    """

    array = np.asarray(array)
    elements = np.where(np.logical_and((array > low_bound), (array < up_bound)))[0]
    try:
        idx = np.min(elements)
        return idx, array[idx]
    except:
        return False, False


def find_last_within(array, low_bound, up_bound):
    """
    Find the first element from an array in a certain interval

    Parameters
    ----------
    array: ndarray
        The 1D array to look in for the nearest value.
    low_bound: float
        the lower boundary of the search interval
    up_bound: float
        the upper boundary of the search interval

    Returns
    -------
    idx: int
      The index of the element in the array that is nearest to the given value.
    value: float
      The value of the array in the found index.
    """

    array = np.asarray(array)
    elements = np.where(np.logical_and((array > low_bound), (array < up_bound)))[0]
    try:
        idx = np.max(elements)
        return idx, array[idx]
    except:
        return False, False


def find_all_within(array, low_bound, up_bound):
    """
    Find the first element from an array in a certain interval

    Parameters
    ----------
    array: ndarray
        The 1D array to look in for the values.
    low_bound: float
        the lower boundary of the search interval
    up_bound: float
        the upper boundary of the search interval

    Returns
    -------
    idx: int
      The index of the elements in the array that are cotained by the given bounds.
    value: float
      The values of the array in the found indexes.
    """

    array = np.asarray(array)
    elements = np.where(np.logical_and((array > low_bound), (array < up_bound)))[0]
    try:
        idx = np.max(elements)
        return idx, array[idx]
    except:
        return False, False


def flatten_list(ls):
    flat_list = [element for sublist in ls for element in sublist]
    return flat_list



def butter_bandpass_filter(data, band_id, sfreq=1200, order=3):
    l_freq, h_freq = get_freq_band(band_id=band_id)
    b, a = butter(N=order, Wn=[l_freq, h_freq], fs=sfreq, btype='band')
    y = lfilter(b, a, data)
    return y


def get_freq_band(band_id):
    '''
    :param band_id: str ('Delta/Theta/Alpha/Beta/Gamma
        String determining the frequency bands to get.

    :return: l_freq: int
        Lower edge of frequency band.
    :return: h_freq: int
        High edge of frequency band.
    '''
    if type(band_id) == str:

        # Get multiple frequency bands
        bands = band_id.split('_')
        l_freqs = []
        h_freqs = []

        for band in bands:
            if band == 'Delta':
                l_freq = 1
                h_freq = 4
            elif band == 'Theta':
                l_freq = 4
                h_freq = 8
            elif band == 'Alpha':
                l_freq = 8
                h_freq = 12
            elif band == 'Beta':
                l_freq = 12
                h_freq = 30
            elif band == 'Gamma':
                l_freq = 30
                h_freq = 40
            elif band == 'HGamma':
                l_freq = 40
                h_freq = 50
            elif band == 'XHGamma':
                l_freq = 50
                h_freq = 100
            elif band == 'Broad':
                l_freq = 0.5
                h_freq = 100
            else:
                raise ValueError(f'Band id {band_id} not recognized.')

            l_freqs.append(l_freq)
            h_freqs.append(h_freq)

        l_freq = np.min(l_freqs)
        h_freq = np.max(h_freqs)

    elif type(band_id) == tuple:
        l_freq = band_id[0]
        h_freq = band_id[1]

    elif band_id == None:

        l_freq = None
        h_freq = None

    return l_freq, h_freq


def get_time_lims(subject, epoch_id, plot_edge=0, map=None):
    '''
    :param epoch_id: str
        String with the name of the epochs to select.
    :param map: dict
        Dictionary of dictionaries indicating the times associated to each type of epoch id.
        Keys should be 'fix', 'sac', and within those keys, a dictionary with keys 'tmin', 'tmax', 'plot_xlim' with their corresponding values.

    :return: tmin: float
        time corresponding to time start of the epochs.
    :return: tmax: float
        time corresponding to time end of the epochs.
    :return: plot_xlim: tuple of float
        time start and end to plot.

    '''
    if map and epoch_id in map.keys():
        tmin = map[epoch_id]['tmin']
        tmax = map[epoch_id]['tmax']
        plot_xlim = map[epoch_id]['plot_xlim']

    else:
        try:
            map = dict(DA={'tmin': 0, 'tmax': subject.exp_times['da_end'] - subject.exp_times['da_start'], 'plot_xlim': [-1, 5.5]},
                       CF={'tmin': 0, 'tmax': subject.exp_times['cf_end'] - subject.exp_times['cf_start'], 'plot_xlim': [-1, 5.5]},
                       baseline={'tmin': 0, 'tmax': subject.exp_times['cf_end'] - subject.exp_times['cf_start'], 'plot_xlim': [-1, 5.5]},
                       sac={'tmin': -0.2, 'tmax': 0.5, 'plot_xlim': [-0.2 + plot_edge, 0.5 - plot_edge]},
                       fix={'tmin': -0.2, 'tmax': 0.5, 'plot_xlim': [-0.2 + plot_edge, 0.5 - plot_edge]},
                       pur={'tmin': -0.2, 'tmax': 0.5, 'plot_xlim': [-0.2 + plot_edge, 0.5 - plot_edge]},
                       )

            if 'fix' in epoch_id:
                tmin = map['fix']['tmin']
                tmax = map['fix']['tmax']
                plot_xlim = map['fix']['plot_xlim']
            elif 'sac' in epoch_id:
                tmin = map['sac']['tmin']
                tmax = map['sac']['tmax']
                plot_xlim = map['sac']['plot_xlim']
            else:
                for key in map.keys():
                    if key in epoch_id:
                        tmin = map[key]['tmin']
                        tmax = map[key]['tmax']
                        plot_xlim = map[key]['plot_xlim']
                        break
                print(f'Using default time values for {epoch_id}: tmin:{tmin}, tmax: {tmax}, plot lims: {plot_xlim}')
        except:
            raise ValueError(f'Epoch id {epoch_id} not in default map keys {map.keys()}.')

    return tmin, tmax, plot_xlim


def get_baseline_duration(epoch_id, tmin, tmax, plot_edge=None, map=None):
    # Baseline duration
    if map and epoch_id in map.keys():
        baseline = (map[epoch_id]['baseline'][0], map[epoch_id]['baseline'][1])
        plot_baseline = (map[epoch_id]['plot_baseline'][0], map[epoch_id]['plot_baseline'][1])

    else:
        if 'sac' in epoch_id:
            baseline = (tmin, 0)
        elif 'fix' in epoch_id:
            baseline = (tmin, -0.05)
        else:
            print(f'Using default baseline from tmin: {tmin} to 0')
            baseline = (tmin, 0)

        if plot_edge:
            plot_baseline = (baseline[0] + plot_edge, baseline[1])
        else:
            plot_baseline = baseline

    # Correct incongruencies
    if baseline[0] < tmin:
        baseline = (tmin, baseline[1])
    if baseline[1] > tmax:
        baseline = (baseline[1], tmax)
    if baseline[0] > baseline[1]:
        print('Baseline start is greater than end. Setting to (tmin)')
        baseline = (tmin, tmin)

    if plot_baseline[0] < tmin:
        plot_baseline = (tmin, plot_baseline[1])
    if plot_baseline[1] > tmax:
        plot_baseline = (plot_baseline[0], tmax)
    if plot_baseline[0] > plot_baseline[1]:
        print('Plot_baseline start is greater than end. Setting to (tmin)')
        plot_baseline = (tmin, tmin)

    return baseline, plot_baseline



def pick_chs(chs_id, info):
    '''
    :param chs_id: 'mag'/'LR'/'parietal/occipital/'frontal'/sac_chs/parietal+'
        String identifying the channels to pick. can be any region or combination of regions selecting also hemisphere.
        Regions and hemispheres should be indicated separated by "_" (parietal_occipital_L).
    :param info: class attribute
        info attribute from the evoked data.
    :return: picks: list
        List of chosen channel names.
    '''

    if chs_id == 'mag':
        picks = [ch_name for ch_name, ch_type in zip(info.ch_names,  info.get_channel_types()) if ch_type == 'mag']

    else:
        ids = chs_id.split('_')
        if 'mag' in ids:
            ids.remove('mag')
        picks = []
        for id in ids:
            if id == 'parietal':
                picks += [ch_name for ch_name, ch_type in zip(info.ch_names,  info.get_channel_types()) if (ch_type == 'mag' and ch_name.startswith('P'))]
            elif id == 'occipital':
                picks += [ch_name for ch_name, ch_type in zip(info.ch_names,  info.get_channel_types()) if (ch_type == 'mag' and ch_name.startswith('O'))]
            elif id == 'frontal':
                picks += [ch_name for ch_name, ch_type in zip(info.ch_names,  info.get_channel_types()) if (ch_type == 'mag' and ch_name.startswith('F'))]
            elif id == 'temporal':
                picks += [ch_name for ch_name, ch_type in zip(info.ch_names,  info.get_channel_types()) if (ch_type == 'mag' and ch_name.startswith('T'))]

            # Subset from picked chanels
            elif id == 'L':
                picks = [ch_name for ch_name, ch_type in zip(info.ch_names,  info.get_channel_types()) if (ch_type == 'mag' and ch_name[1] == 'L')]
            elif id == 'R':
                picks = [ch_name for ch_name, ch_type in zip(info.ch_names,  info.get_channel_types()) if (ch_type == 'mag' and ch_name[1] == 'R')]

            # Subset from picked chanels
            elif id == 'z':
                picks = [ch_name for ch_name, ch_type in zip(info.ch_names, info.get_channel_types()) if
                         (ch_type == 'mag' and ch_name.endswith('[Z]'))]
            elif id == 'x':
                picks = [ch_name for ch_name, ch_type in zip(info.ch_names, info.get_channel_types()) if
                         (ch_type == 'mag' and ch_name.endswith('[X]'))]
            elif id == 'y':
                picks = [ch_name for ch_name, ch_type in zip(info.ch_names, info.get_channel_types()) if
                         (ch_type == 'mag' and ch_name.endswith('[Y]'))]

    return picks


def get_stim_channel_names(meg_data):
    """
    Get all channel names of type 'stim' from MEG data.

    Parameters
    ----------
    meg_data : instance of mne.io.Raw
        The raw MEG data.

    Returns
    -------
    stim_channels : list
        List of stimulus channel names.
    """
    # Get channel names and types
    ch_names = meg_data.ch_names
    ch_types = meg_data.get_channel_types()

    # Find stimulus channels
    stim_channels = [ch_names[i] for i, ch_type in enumerate(ch_types) if ch_type == 'stim']

    return stim_channels
