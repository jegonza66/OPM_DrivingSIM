import copy
import numpy as np
import pandas as pd
import os
import math
import mne
import scipy.signal as sgn
import functions_general
import paths
import save


def reescale_et_channels(meg_gazex_data_raw, meg_gazey_data_raw, minvoltage=-5, maxvoltage=5, minrange=-0.2, maxrange=1.2,
                         screenright=1919, screenleft=0, screentop=0, screenbottom=1079):
    """
    Reescale Gaze data from MEG Eye Tracker channels to correct digital to analog conversion.

    Parameters
    ----------
    meg_gazex_data_raw: ndarray
        Raw gaze x data to rescale
    meg_gazey_data_raw: ndarray
        Raw gaze x data to rescale
    minvoltage: int
        The minimum voltage value from the digital-analog conversion.
        Default to -5 from analog.ini analog_dac_range.
    maxvoltage: int
        The maximum voltage value from the digital-analog conversion.
        Default to 5 from analog.ini analog_dac_range.
    minrange: int
        The minimum gaze position tracked by the eye tracker outside the screen.
        Default to -0.2 from analog.ini analog_x_range to allow for +/- 20% outside display
    maxrange: int
        The maximum gaze position tracked by the eye tracker outside the screen.
        Default to 1.2 from analog.ini analog_x_range to allow for +/- 20% outside display
    screenright: int
        Pixel number of the right side of the screen. Default to 1919.
    screenleft: int
        Pixel number of the left side of the screen. Default to 0.
    screentop: int
        Pixel number of the top side of the screen. Default to 0.
    screenbottom: int
        Pixel number of the bottom side of the screen. Default to 1079.

    Returns
    -------
    meg_gazex_data_scaled: ndarray
        The scaled gaze x data
    meg_gazey_data_scaled: ndarray
        The scaled gaze y data
    """

    print('Rescaling')
    # Scale
    R_h = (meg_gazex_data_raw - minvoltage) / (maxvoltage - minvoltage)  # voltage range proportion
    S_h = R_h * (maxrange - minrange) + minrange  # proportion of screen width or height
    R_v = (meg_gazey_data_raw - minvoltage) / (maxvoltage - minvoltage)
    S_v = R_v * (maxrange - minrange) + minrange
    meg_gazex_data_scaled = S_h * (screenright - screenleft + 1) + screenleft
    meg_gazey_data_scaled = S_v * (screenbottom - screentop + 1) + screentop

    return meg_gazex_data_scaled, meg_gazey_data_scaled


def blinks_to_nan(meg_data, et_channels_meg):
    print('Removing blinks using MEG annotations')

    # Copy data to avoid modifying in place
    meg_gazex_data_clean = et_channels_meg[0]
    meg_gazey_data_clean = et_channels_meg[1]
    meg_pupils_data_clean = et_channels_meg[2]

    sfreq = meg_data.info['sfreq']
    n_samples = len(meg_gazex_data_clean)

    # Find blink annotations
    if hasattr(meg_data, 'annotations') and meg_data.annotations is not None:
        for onset, duration, desc in zip(meg_data.annotations.onset, meg_data.annotations.duration, meg_data.annotations.description):
            if desc.lower() == 'blink':
                # Onsets from annotations start at meg first time due to something in preprocessing
                start_sample = int((onset - meg_data.first_time) * sfreq)
                end_sample = int((onset + duration - meg_data.first_time) * sfreq)
                # Clip to data range
                start_sample = max(0, start_sample)
                end_sample = min(n_samples, end_sample)
                meg_gazex_data_clean[start_sample:end_sample] = np.nan
                meg_gazey_data_clean[start_sample:end_sample] = np.nan
                meg_pupils_data_clean[start_sample:end_sample] = np.nan
    else:
        print('No annotations found in MEG data.')

    return et_channels_meg


def DAC_samples(et_channels_meg, exp_info, sfreq):

    print('Compensating DAC delay')
    time_delay = exp_info.DAC_delay
    samples_delay = int(round(time_delay / 1000 * sfreq, 0))

    for i in range(len(et_channels_meg)):
        et_channels_meg[i] = np.concatenate((et_channels_meg[i][samples_delay:], np.zeros(samples_delay)))

    # Get separate data from et channels
    meg_gazex_data_raw = et_channels_meg[0]
    meg_gazey_data_raw = et_channels_meg[1]
    meg_pupils_data_raw = et_channels_meg[2]

    return meg_gazex_data_raw, meg_gazey_data_raw, meg_pupils_data_raw


def remove_annotations(meg_data, subject, exp_info):
    """
    Remove annotations with names containing 'saccade' and 'fixation' from MEG data.

    Parameters
    ----------
    meg_data : instance of mne.io.Raw
        The raw MEG data containing annotations.
    subject : instance of subject class
        Subject object (for consistency with function signature).
    exp_info : instance of exp_info class
        Experiment information object (for consistency with function signature).

    Returns
    -------
    meg_data : instance of mne.io.Raw
        The MEG data with saccade and fixation annotations removed.
    """
    print('Removing saccade and fixation annotations from MEG data')

    if hasattr(meg_data, 'annotations') and meg_data.annotations is not None:
        # Get current annotations
        onsets = meg_data.annotations.onset
        durations = meg_data.annotations.duration
        descriptions = meg_data.annotations.description

        # Find indices of annotations to keep (not saccade or fixation)
        keep_indices = []
        removed_count = 0

        for i, desc in enumerate(descriptions):
            desc_lower = desc.lower()
            if 'sac' not in desc_lower and 'fix' not in desc_lower:
                keep_indices.append(i)
            else:
                removed_count += 1

        if removed_count > 0:
            print(f'Removed {removed_count} saccade/fixation annotations')

            # Create new annotations with only the kept ones
            if keep_indices:
                new_onsets = onsets[keep_indices]
                new_durations = durations[keep_indices]
                new_descriptions = descriptions[keep_indices]

                new_annotations = mne.Annotations(onset=new_onsets,
                                                duration=new_durations,
                                                description=new_descriptions)
                meg_data.set_annotations(new_annotations)
            else:
                # No annotations left, set to None
                meg_data.set_annotations(None)
        else:
            print('No saccade/fixation annotations found to remove')
    else:
        print('No annotations found in MEG data')

    return meg_data


def fixations_saccades_detection(meg_data, et_channels_meg, subject, exp_info, sac_max_vel=1500, fix_max_amp=1.5,
                                 screen_resolution=1920, force_run=False):

    out_fname = f'Fix_Sac_detection_{subject.subject_id}.tsv'
    out_folder = paths.processed_path + subject.subject_id + '/Sac-Fix_detection/'

    meg_gazex_data_clean = et_channels_meg[0]
    meg_gazey_data_clean = et_channels_meg[1]

    if not force_run:
        try:
            # Load pre run saccades and fixation detection
            sac_fix = pd.read_csv(out_folder + out_fname, sep='\t')
            print('\nSaccades and fixations loaded')
        except:
            force_run = True

    if force_run:
            # If not pre run data, run
            print('\nRunning saccades and fixations detection')

            # Define data to save to excel file needed to run the saccades detection program Remodnav
            eye_data = {'x': meg_gazex_data_clean, 'y': meg_gazey_data_clean}
            df = pd.DataFrame(eye_data)

            # Remodnav parameters
            fname = f'eye_data_{subject.subject_id}.csv'
            px2deg = math.degrees(math.atan2(.5 * exp_info.screen_size, subject.screen_distance)) / (.5 * screen_resolution)
            sfreq = meg_data.info['sfreq']

            # Save csv file
            df.to_csv(fname, sep='\t', header=False, index=False)

            # Run Remodnav not considering pursuit class and min fixations 100 ms
            command = (f'remodnav {fname} {out_fname} {px2deg} {sfreq} --savgol-length {0.0195} --min-pursuit-duration {0.1} '
                       f'--max-pso-duration {0.0} --min-saccade-duration {0.01} --min-fixation-duration {0.05} --max-vel {5000} '
                       f'--pursuit-velthresh {1.5}')
            os.system(command)

            # Read results file with detections
            sac_fix = pd.read_csv(out_fname, sep='\t')

            # Move eye data, detections file and image to subject results directory
            os.makedirs(out_folder, exist_ok=True)
            # Move et data file
            os.replace(fname, out_folder + fname)
            # Move results file
            os.replace(out_fname, out_folder + out_fname)
            # Move results image
            out_fname = out_fname.replace('tsv', 'png')
            os.replace(out_fname, out_folder + out_fname)

    # Get saccades and fixations
    saccades_all = copy.copy(sac_fix.loc[(sac_fix['label'] == 'SACC') | (sac_fix['label'] == 'ISAC')])
    fixations_all = copy.copy(sac_fix.loc[sac_fix['label'] == 'FIXA'])
    pursuits_all = copy.copy(sac_fix.loc[sac_fix['label'] == 'PURS'])

    # Drop saccades and fixations based on conditions
    print(f'Dropping saccades with average vel > {sac_max_vel}, and fixations with amplitude > {fix_max_amp}')
    fixations = copy.copy(fixations_all[(fixations_all['amp'] <= fix_max_amp)])
    saccades = copy.copy(saccades_all[saccades_all['peak_vel'] <= sac_max_vel])
    pursuits = copy.copy(pursuits_all)
    print(f'Kept {len(fixations)} out of {len(fixations_all)} fixations')
    print(f'Kept {len(saccades)} out of {len(saccades_all)} saccades')
    print(f'Total {len(pursuits)} pursuit events')

    return fixations, saccades, pursuits


def saccades_classification(subject, saccades, meg_data):
    print('\nClassifying saccades')

    # Find trial onset from MEG annotation 'drive'
    drive_idx = np.where(meg_data.annotations.description == 'drive')[0]
    if len(drive_idx) == 0:
        raise ValueError("No 'drive' annotation found for trial onset.")
    trial_onset = meg_data.annotations.onset[drive_idx[0]]  # Not withdrawing meg first time because fixations onset from remodnav starts from 0

    # Prepare lists for new columns
    n_sacs = []
    sac_delay = []
    sac_deg = []
    sac_dir = []

    for i, (_, saccade) in enumerate(saccades.iterrows()):
        # Saccade number (1-based)
        n_sacs.append(i + 1)
        # Delay from trial onset
        delay = saccade['onset'] - trial_onset
        sac_delay.append(delay)
        # Saccade direction (deg)
        x_dif = saccade['end_x'] - saccade['start_x']
        y_dif = saccade['end_y'] - saccade['start_y']
        z = complex(x_dif, y_dif)
        deg = np.angle(z, deg=True)
        sac_deg.append(deg)
        # Direction label
        if -15 < deg < 15:
            dir = 'r'
        elif 75 < deg < 105:
            dir = 'd'
        elif 165 < deg or deg < -165:
            dir = 'l'
        elif -75 > deg > -105:
            dir = 'u'
        else:
            dir = 'none'
        sac_dir.append(dir)

    # Add columns to saccades DataFrame
    saccades['n_sac'] = n_sacs
    saccades['delay'] = sac_delay
    saccades['deg'] = sac_deg
    saccades['dir'] = sac_dir

    # Drop saccades with blinks in between
    if hasattr(meg_data, 'annotations') and meg_data.annotations is not None:
        saccades_to_keep = []
        for idx, (_, saccade) in enumerate(saccades.iterrows()):
            has_blink = False
            sac_start = saccade['onset'] + meg_data.first_time
            sac_end = sac_start + saccade['duration']

            for onset, duration, desc in zip(meg_data.annotations.onset, meg_data.annotations.duration, meg_data.annotations.description):
                if desc.lower() == 'blink':
                    blink_start = onset
                    blink_end = blink_start + duration
                    # Check if blink overlaps with saccade
                    if not (blink_end <= sac_start or blink_start >= sac_end):
                        has_blink = True
                        break

            if not has_blink:
                saccades_to_keep.append(idx)

        initial_count = len(saccades)
        saccades = saccades.iloc[saccades_to_keep]
        print(f'Dropped {initial_count - len(saccades)} saccades with blinks, kept {len(saccades)}')

    # Set types
    saccades = saccades.astype({'n_sac': 'Int64', 'delay': float, 'deg': float, 'dir': str})

    # Add vs fixations data to meg_data annotations
    onset_list = saccades['onset'].values + meg_data.first_time  # Adjust onset to start from 0
    duration_list = saccades['duration'].values
    description_list = ['sac_' + str(val) for val in saccades['dir']]
    meg_data.annotations.description = np.concatenate((meg_data.annotations.description, np.array(description_list)))
    meg_data.annotations.onset = np.concatenate((meg_data.annotations.onset - meg_data.first_time, np.array(onset_list)))  # Correct original annotations timeing only here
    meg_data.annotations.duration = np.concatenate((meg_data.annotations.duration, np.array(duration_list)))

    # Sort annotations by onset
    meg_annot = np.array([meg_data.annotations.onset, meg_data.annotations.duration, meg_data.annotations.description])
    meg_annot = meg_annot[:, meg_annot[0].astype(float).argsort()]
    meg_data.annotations.onset = meg_annot[0].astype(float)
    meg_data.annotations.duration = meg_annot[1].astype(float)
    meg_data.annotations.description = meg_annot[2]

    return saccades, subject


def fixations_classification(df, saccades, meg_data, et_channels_meg, title):
    print(f'\nClassifying {title}')

    # Find trial onset from MEG annotation 'drive'
    drive_idx = np.where(meg_data.annotations.description == 'drive')[0]
    if len(drive_idx) == 0:
        raise ValueError("No 'drive' annotation found for trial onset.")
    trial_onset = meg_data.annotations.onset[drive_idx[0]]  # Not withdrawing meg first time because fixations onset from remodnav starts from 0

    times = meg_data.times
    sacc_thresh = 0.01  # 10 ms

    mean_x = []
    mean_y = []
    pupil_size = []
    prev_sac = []
    next_sac = []
    n_evts = []
    evt_delay = []

    for i, (_, row) in enumerate(df.iterrows()):
        onset = row['onset']
        dur = row['duration']

        # Previous and next saccades
        try:
            sac0 = saccades.loc[(saccades['onset'] + saccades['duration'] > onset - sacc_thresh) & (
                        saccades['onset'] + saccades['duration'] < onset + sacc_thresh)].index.values[-1]
        except:
            sac0 = None

        try:
            sac1 = saccades.loc[(saccades['onset'] > onset + dur - sacc_thresh) & (
                        saccades['onset'] < onset + dur + sacc_thresh)].index.values[0]
        except:
            sac1 = None

        prev_sac.append(sac0)
        next_sac.append(sac1)

        # n_fixs: sequential number after trial onset
        n_evts.append(i + 1)

        # fix_delay: time since trial onset
        evt_delay.append(onset - trial_onset)

        # Average pupil size, x and y position
        fix_time_idx = np.where(np.logical_and(onset < times, times < onset + dur))[0]

        gazex_data = et_channels_meg[0][fix_time_idx]
        gazey_data = et_channels_meg[1][fix_time_idx]
        pupil_data = et_channels_meg[2][fix_time_idx]
        pupil_size.append(np.nanmean(pupil_data))
        mean_x.append(np.nanmean(gazex_data))
        mean_y.append(np.nanmean(gazey_data))
        print("\rProgress: {}%".format(int((i + 1) * 100 / len(df))), end='')

    # Add columns
    df['prev_sac'] = prev_sac
    df['next_sac'] = next_sac
    df['n_fix'] = n_evts
    df['delay'] = evt_delay
    df['mean_x'] = mean_x
    df['mean_y'] = mean_y
    df['pupil'] = pupil_size

    # # Remove events with no previous or next saccade (i.e., coming from or going to blink)
    # fixations.dropna(subset=['prev_sac', 'next_sac'], inplace=True)
    df = df.astype({'mean_x': float, 'mean_y': float, 'pupil': float, 'prev_sac': 'Int64', 'next_sac': 'Int64', 'n_fix': 'Int64', 'delay': float})

    print(f'\nKept {len(df)} {title} with previous and next saccade')

    # Add vs fixations data to meg_data annotations
    onset_list = df['onset'].values + meg_data.first_time  # Adjust onset to start from 0
    description_list = [title[:3]] * len(df)
    duration_list = df['duration'].values
    meg_data.annotations.description = np.concatenate((meg_data.annotations.description, np.array(description_list)))
    meg_data.annotations.onset = np.concatenate((meg_data.annotations.onset, np.array(onset_list)))
    meg_data.annotations.duration = np.concatenate((meg_data.annotations.duration, np.array(duration_list)))

    # Sort annotations by onset
    meg_annot = np.array([meg_data.annotations.onset, meg_data.annotations.duration, meg_data.annotations.description])
    meg_annot = meg_annot[:, meg_annot[0].astype(float).argsort()]
    meg_data.annotations.onset = meg_annot[0].astype(float)
    meg_data.annotations.duration = meg_annot[1].astype(float)
    meg_data.annotations.description = meg_annot[2]

    return df


def interpolate_bad_channels(subject_id, meg_data, exp_info):

    # Interpolate bad channels by orientation (X, Y, Z)
    bads = list(set(exp_info.bad_channels.get(subject_id, []) + meg_data.info.get('bads', [])))
    if bads:
        print(f'Interpolating bad channels for subject {subject_id}: {bads}')
        ch_names = meg_data.info['ch_names']
        bads_by_orient = {'X': [], 'Y': [], 'Z': []}
        for ch in bads:
            if ch.endswith('[X]'):
                bads_by_orient['X'].append(ch)
            elif ch.endswith('[Y]'):
                bads_by_orient['Y'].append(ch)
            elif ch.endswith('[Z]'):
                bads_by_orient['Z'].append(ch)

        # Interpolate for each orientation separately
        for orient, bad_list in bads_by_orient.items():
            if not bad_list:
                continue
            orient_chs = [ch for ch in ch_names if ch.endswith(f'[{orient}]')]
            # Create a Raw object with only this orientation
            raw_orient = meg_data.copy().pick_channels(orient_chs)
            raw_orient.info['bads'] = bad_list
            raw_orient.interpolate_bads(reset_bads=True)
            # Copy interpolated data back to original
            for idx, ch in enumerate(orient_chs):
                orig_idx = meg_data.ch_names.index(ch)
                meg_data._data[orig_idx] = raw_orient._data[idx]
        meg_data.info['bads'] = []  # Optionally clear bads after interpolation
    else:
        print(f'No bad channels to interpolate for subject {subject_id}')

    return meg_data


def add_et_channels(meg_data, et_channels_meg, exp_info, subject_id):
    #---------------- Add scaled data to meg data ----------------#
    print('\nSaving scaled et data to meg raw data structure')
    # make new raw structure from et channels only
    raw_et = mne.io.RawArray(et_channels_meg, meg_data.copy().pick(exp_info.et_channel_names[subject_id]).info)
    # change channel names
    for ch_name, new_name in zip(raw_et.ch_names, ['ET_x', 'ET_y', 'ET_pupils']):
        raw_et.rename_channels({ch_name: new_name})

    # save to original raw structure (requires to load data)
    print('Loading MEG data')
    meg_data.load_data()
    print('Adding new ET channels')
    meg_data.add_channels([raw_et], force_update_info=True)

    # Drop the et channels
    meg_data.drop_channels(exp_info.et_channel_names[subject_id])

    return meg_data


def overwrite_et_channels(meg_data, et_channels_meg, exp_info, subject_id):
    #---------------- Add scaled data to meg data ----------------#
    print('\nSaving scaled et data to meg raw data structure')

    # Get et channels idx
    et_ch_idx = mne.pick_channels(meg_data.info['ch_names'], exp_info.et_channel_names[subject_id])

    # Overwrite data in et channels idx for scaled data
    meg_data.load_data()
    meg_data._data[et_ch_idx, :] = et_channels_meg

    return meg_data


def set_digitlization(subject, meg_data):

    # Load digitalization file
    dig_path = paths().opt_path()
    dig_path_subject = dig_path + subject.subject_id
    dig_filepath = dig_path_subject + '/Model_Mesh_5m_headers.pos'
    pos = pd.read_table(dig_filepath, index_col=0)

    # Get fiducials from dig
    nasion = pos.loc[pos.index == 'nasion ']
    lpa = pos.loc[pos.index == 'left ']
    rpa = pos.loc[pos.index == 'right ']

    # Get head points
    pos.drop(['nasion ', 'left ', 'right '], inplace=True)
    pos_array = pos.to_numpy()

    # Make montage
    dig_montage = mne.channels.make_dig_montage(nasion=nasion.values.ravel(), lpa=lpa.values.ravel(),
                                                rpa=rpa.values.ravel(), hsp=pos_array, coord_frame='unknown')

    # Make info object
    meg_data.info.set_montage(montage=dig_montage)

    return meg_data



def save(subject_id, meg_data, fixations, saccades, pursuits, task):
    """
    Save preprocessed data
    :param meg_data:
    :param subject:
    :param bh_data:
    :param fixations:
    :param saccades:
    """

    print('Saving preprocessed data')
    # Path
    processed_data_path = paths.processed_path
    processed_save_path = processed_data_path + subject_id + '/'
    os.makedirs(processed_save_path, exist_ok=True)

    # Save fixations ans saccades
    fixations.to_csv(processed_save_path + 'fixations.csv')
    saccades.to_csv(processed_save_path + 'saccades.csv')
    pursuits.to_csv(processed_save_path + 'pursuits.csv')

    # Save MEG
    preproc_meg_data_fname = f'{task}_{subject_id}_meg.fif'
    meg_data.save(processed_save_path + preproc_meg_data_fname, overwrite=True)

    print(f'Preprocessed data saved to {processed_save_path}')