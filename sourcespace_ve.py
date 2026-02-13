# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 20:58:58 2025

@author: lpxaj7
"""
import mne
import numpy as np
from scipy.signal import hilbert
import functions_general
import load
import paths


def _find_target_vertex_index(mni_coord, src, subject_code, source_params):
    """
    Helper function to find target vertex index for a given MNI coordinate.

    Parameters
    ----------
    mni_coord : array-like, shape (3,)
        MNI coordinates [x, y, z] in mm.
    src : mne.SourceSpaces
        Source space for the subject.
    subject_code : str
        Subject code ('fsaverage' for template, subject_id for individual).
    source_params : dict, optional
        Source space parameters (required for individual subjects).

    Returns
    -------
    target_voxel_idx : int
        Index of the target vertex in the source space.
    actual_pos : array, shape (3,)
        Actual position of the target vertex in mm.
    """
    import os

    if subject_code == 'fsaverage':
        # For fsaverage, coordinates are already in MNI space
        print("Using fsaverage template - coordinates are in MNI space")

        # Find target voxel in fsaverage source space
        if src[0]['type'] == 'surf':  # Surface source space
            lh_pos = src[0]['rr'][src[0]['vertno']] * 1000  # Convert to mm
            rh_pos = src[1]['rr'][src[1]['vertno']] * 1000
            all_pos = np.vstack([lh_pos, rh_pos])
        else:  # Volume source space
            all_pos = src[0]['rr'][src[0]['vertno']] * 1000

        # Find closest vertex to target MNI coordinate
        distances = np.linalg.norm(all_pos - mni_coord, axis=1)
        target_voxel_idx = np.argmin(distances)
        actual_pos = all_pos[target_voxel_idx]

        print(f"Target MNI coordinate: {mni_coord}")
        print(f"Closest vertex at: {actual_pos} (distance: {distances[target_voxel_idx]:.1f} mm)")

    else:
        # For individual subjects, morph to fsaverage for coordinate finding
        print(f"Using individual subject {subject_code} - will morph to fsaverage space for coordinate matching")

        subjects_dir = os.environ.get('SUBJECTS_DIR')
        if subjects_dir is None:
            raise ValueError("SUBJECTS_DIR environment variable not set")

        # Load fsaverage source space to find target coordinate
        source_path_fsaverage = paths.sources_path + 'fsaverage'
        src_fsaverage = load.source_model(sources_path_subject=source_path_fsaverage, subject_code='fsaverage', source_params=source_params)

        # Find target coordinate in fsaverage space
        if src_fsaverage[0]['type'] == 'surf':  # Surface source space
            lh_pos_fsave = src_fsaverage[0]['rr'][src_fsaverage[0]['vertno']] * 1000
            rh_pos_fsave = src_fsaverage[1]['rr'][src_fsaverage[1]['vertno']] * 1000
            all_pos_fsave = np.vstack([lh_pos_fsave, rh_pos_fsave])
        else:  # Volume source space
            all_pos_fsave = src_fsaverage[0]['rr'][src_fsaverage[0]['vertno']] * 1000

        distances = np.linalg.norm(all_pos_fsave - mni_coord, axis=1)
        target_idx_fsave = np.argmin(distances)

        print(f"Target MNI coordinate: {mni_coord}")
        print(f"Closest fsaverage vertex at: {all_pos_fsave[target_idx_fsave]} (distance: {distances[target_idx_fsave]:.1f} mm)")

        # Create morph from fsaverage to subject
        morph_fsave_to_subj = mne.compute_source_morph(
            src_fsaverage, subject_from='fsaverage', subject_to=subject_code,
            src_to=src, subjects_dir=subjects_dir
        )

        # Use dummy STC approach to find corresponding subject vertex
        n_vertices_fsave = len(src_fsaverage[0]['vertno']) + len(src_fsaverage[1]['vertno'])
        dummy_data = np.zeros((n_vertices_fsave, 1))
        dummy_data[target_idx_fsave, 0] = 1.0

        if src_fsaverage[0]['type'] == 'surf':
            dummy_stc_fsave = mne.SourceEstimate(
                data=dummy_data,
                vertices=[src_fsaverage[0]['vertno'], src_fsaverage[1]['vertno']],
                tmin=0, tstep=1.0
            )
        else:
            dummy_stc_fsave = mne.VolSourceEstimate(
                data=dummy_data,
                vertices=src_fsaverage[0]['vertno'],
                tmin=0, tstep=1.0
            )

        # Morph to subject space to find corresponding vertex
        dummy_stc_subj = morph_fsave_to_subj.apply(dummy_stc_fsave)
        target_voxel_idx = np.argmax(dummy_stc_subj.data[:, 0])

        # Get actual position in subject space
        if src[0]['type'] == 'surf':
            lh_pos = src[0]['rr'][src[0]['vertno']] * 1000
            rh_pos = src[1]['rr'][src[1]['vertno']] * 1000
            all_pos = np.vstack([lh_pos, rh_pos])
        else:
            all_pos = src[0]['rr'][src[0]['vertno']] * 1000

        actual_pos = all_pos[target_voxel_idx]
        print(f"Corresponding subject vertex at: {actual_pos}")

    return target_voxel_idx, actual_pos


def _ve_from_evoked(evoked, filters, chs_id, idx, pick_ori):
    # Use pre-computed filters instead of recomputing them
    weights = filters["weights"]
    picks = functions_general.pick_chs(chs_id=chs_id, info=evoked.info)
    data_matrix = evoked.get_data(picks=picks)

    # Combine orientations if needed
    if not pick_ori:
        ve = np.zeros((3, data_matrix.shape[1]))

        base_idx = idx - (idx % 3)
        peak_weights_x = weights[base_idx, :]
        peak_weights_y = weights[base_idx + 1, :]
        peak_weights_z = weights[base_idx + 2, :]

        ve[0, :] = peak_weights_x.T @ data_matrix
        ve[1, :] = peak_weights_y.T @ data_matrix
        ve[2, :] = peak_weights_z.T @ data_matrix

        comb = ve[0, :] ** 2
        comb += ve[1, :] ** 2
        comb += ve[2, :] ** 2
        comb = np.sqrt(comb)
        ve = comb

    else:
        peak_weights = weights[idx, :]
        ve = peak_weights.T @ data_matrix

    return ve


# Directly apply beamformer weights for target voxel only
def _ve_from_epoch(epoch, filters, chs_id, idx, pick_ori):
    weights = filters["weights"]
    picks = functions_general.pick_chs(chs_id=chs_id, info=epoch.info)  # Adjust chs_id as needed
    data_matrix = epoch.get_data(picks=picks)

    # Combine orientations if needed
    if not pick_ori:

        ve_trials = np.zeros((data_matrix.shape[0], 3, data_matrix.shape[2]))

        base_idx = idx - (idx % 3)
        target_weights_x = weights[base_idx, :]
        target_weights_y = weights[base_idx + 1, :]
        target_weights_z = weights[base_idx + 2, :]

        for tr_ind, mat in enumerate(data_matrix):
            ve_trials[tr_ind, 0, :] = target_weights_x.T @ mat
            ve_trials[tr_ind, 1, :] = target_weights_y.T @ mat
            ve_trials[tr_ind, 2, :] = target_weights_z.T @ mat

        comb = ve_trials[:, 0, :] ** 2
        comb += ve_trials[:, 1, :] ** 2
        comb += ve_trials[:, 2, :] ** 2
        comb = np.sqrt(comb)
        ve_trials = comb

    else:
        target_weights = weights[idx, :]  # Get weights for target voxel only
        ve_trials = np.zeros((data_matrix.shape[0], data_matrix.shape[2]))
        for tr_ind, mat in enumerate(data_matrix):
            ve_trials[tr_ind] = target_weights.T @ mat

    return ve_trials


def apply_baseline_array(ve_data, ve_times, baseline, source_estimation):
    bline_tmin, _ = functions_general.find_nearest(ve_times, baseline[0])
    bline_tmax, _ = functions_general.find_nearest(ve_times, baseline[1])
    if source_estimation == 'epo':
        ve_data = ve_data - np.expand_dims(ve_data.copy()[:, bline_tmin:bline_tmax].mean(axis=-1), axis=-1)
    elif source_estimation == 'evk':
        ve_data = ve_data - np.expand_dims(ve_data.copy()[bline_tmin:bline_tmax].mean(axis=-1), axis=-1)

    return ve_data


def get_peak_ve_epochs(stc, fwd, epochs, filters, chs_id, pick_ori, mode_peak='abs', subject_code='fsaverage'):
    peak_index = stc.get_peak(mode=mode_peak, vert_as_index=True)[0] #vertex index

    # Extract MNI coordinates of the peak location
    src = fwd['src']
    if src[0]['type'] == 'surf':  # Surface source space
        lh_pos = src[0]['rr'][src[0]['vertno']] * 1000  # Convert to mm
        rh_pos = src[1]['rr'][src[1]['vertno']] * 1000
        all_pos = np.vstack([lh_pos, rh_pos])
    else:  # Volume source space
        all_pos = src[0]['rr'][src[0]['vertno']] * 1000

    # Get the actual MNI coordinates of the peak vertex
    peak_mni_coords = all_pos[peak_index]

    ve_dict = {}
    ve_dict["data"] = _ve_from_epoch(epochs, filters, chs_id=chs_id, idx=peak_index, pick_ori=pick_ori)
    ve_dict["fs"] = epochs.info["sfreq"]
    ve_dict['peak_index']=peak_index
    # Add MNI coordinate information
    ve_dict['actual_mni'] = peak_mni_coords
    ve_dict['subject_code'] = subject_code
    return ve_dict


def get_peak_ve_evoked(stc, fwd, evoked, filters, chs_id, pick_ori, mode_peak='abs', subject_code='fsaverage'):
    peak_index = stc.get_peak(mode=mode_peak, vert_as_index=True)[0] #vertex index

    # Extract MNI coordinates of the peak location
    src = fwd['src']
    if src[0]['type'] == 'surf':  # Surface source space
        lh_pos = src[0]['rr'][src[0]['vertno']] * 1000  # Convert to mm
        rh_pos = src[1]['rr'][src[1]['vertno']] * 1000
        all_pos = np.vstack([lh_pos, rh_pos])
    else:  # Volume source space
        all_pos = src[0]['rr'][src[0]['vertno']] * 1000

    # Get the actual MNI coordinates of the peak vertex
    peak_mni_coords = all_pos[peak_index]

    ve_dict = {}
    ve_dict["data"] = _ve_from_evoked(evoked, filters, chs_id=chs_id, idx=peak_index, pick_ori=pick_ori)
    ve_dict["fs"] = evoked.info["sfreq"]
    ve_dict['peak_index']=peak_index
    # Add MNI coordinate information
    ve_dict['actual_mni'] = peak_mni_coords
    ve_dict['subject_code'] = subject_code
    return ve_dict


# % evoked response at peak location
def get_VE_evoked(ve_dict, tmin, source_estimation, pick_ori=None):
    fs = ve_dict["fs"]

    # Step 1: Average across epochs to get the evoked response
    if source_estimation == 'epo':
        averaged_data = np.expand_dims(ve_dict["data"].mean(axis=0), axis=0)  # shape: (1, n_times,)
    elif source_estimation == 'evk':
        averaged_data = np.expand_dims(ve_dict["data"], axis=0)  # shape: (1, n_times)

    # Step 2: Create Info object
    info = mne.create_info(["Peak VE"], sfreq=fs, ch_types="misc")  # or 'misc' if not EEG
    # Step 3: Create Evoked object
    evoked = mne.EvokedArray(averaged_data, info, tmin=tmin)
    return evoked

def compute_hilbert_envelope(ve, fs, tmin, baseline_hilb, source_estimation):
    """

    Parameters
    ----------
    ve : TYPE
        virtual electrode data.
    fs :  FLOAT
       Sampling frequency in Hz.
    params : dict
       Dictionary of parameters. Must contain:
           - 'epoch_onset' : float
               Time of epoch onset in seconds.
           - other optional processing parameters (not specified here).
    baseline_hilb : TYPE
       Baseline value for normalization.

    Returns
    -------
    env : TYPE
        DESCRIPTION.

    """
    baseline_inds = np.int32(
        np.round(
            np.array([baseline_hilb[0] - (tmin),  # -5= epoch onset (epochs len=-5,5))
                      baseline_hilb[1] - (tmin)])
            * fs
        )
    )

    if source_estimation == 'epo':
        h_envs = np.empty((ve.shape[0], ve.shape[1]))

        for tr_ind, VE_trl in enumerate(ve):
            h_envs[tr_ind, :] = np.abs(hilbert(VE_trl))

        baselines = np.mean(h_envs[:, baseline_inds[0]: baseline_inds[1]], axis=1)
        h_envs_bl = (h_envs - baselines[:, None]) / baselines[:, None]
        env = np.mean(h_envs_bl, axis=0)

    elif source_estimation == 'evk':
        h_envs = np.abs(hilbert(ve)).squeeze()
        baselines = np.mean(h_envs[baseline_inds[0]: baseline_inds[1]])
        env = (h_envs - baselines) / baselines

    return env


def get_coordinate_ve_epochs(mni_coord, fwd, epochs, filters, source_params, chs_id, pick_ori, subject_code='fsaverage'):
    """
    Extract virtual electrode data at a specific MNI coordinate.

    Parameters
    ----------
    mni_coord : array-like, shape (3,)
        MNI coordinates [x, y, z] in mm where to extract the virtual electrode.
    fwd : mne.Forward
        Forward model.
    epochs : mne.Epochs
        Epoched data.
    filters : mne.beamformer.Beamformer
        Pre-computed beamformer filters (preferred).
    source_params : dict
        Source space parameters (only used if filters not provided)
    subject_code : str
        Subject code ('fsaverage' for template, subject_id for individual).

    Returns
    -------
    ve_dict : dict
        Dictionary containing virtual electrode data.
    """
    src = fwd['src']

    # Use helper function to find target vertex
    target_voxel_idx, actual_pos = _find_target_vertex_index(mni_coord, src, subject_code, source_params)

    ve_data = _ve_from_epoch(epochs, filters, chs_id=chs_id, idx=target_voxel_idx, pick_ori=pick_ori)

    # Store results
    ve_dict = {}
    ve_dict["data"] = ve_data
    ve_dict["fs"] = epochs.info["sfreq"]
    ve_dict['peak_index'] = target_voxel_idx
    ve_dict['target_mni'] = mni_coord
    ve_dict['actual_mni'] = actual_pos
    ve_dict['subject_code'] = subject_code

    return ve_dict

def get_coordinate_ve_evoked(mni_coord, fwd, evoked, filters, source_params, chs_id, pick_ori, subject_code='fsaverage'):
    """
    Extract virtual electrode data at a specific MNI coordinate.

    Parameters
    ----------
    mni_coord : array-like, shape (3,)
        MNI coordinates [x, y, z] in mm where to extract the virtual electrode.
    fwd : mne.Forward
        Forward model.
    evoked : mne.Evoked
        Epoched data.
    filters : mne.beamformer.Beamformer
        Pre-computed beamformer filters (preferred).
    source_params : dict
        Source space parameters (only used if filters not provided).
    subject_code : str
        Subject code ('fsaverage' for template, subject_id for individual).

    Returns
    -------
    ve_dict : dict
        Dictionary containing virtual electrode data.
    """
    src = fwd['src']

    # Use helper function to find target vertex
    target_voxel_idx, actual_pos = _find_target_vertex_index(mni_coord, src, subject_code, source_params)

    ve_data = _ve_from_evoked(evoked, filters, chs_id=chs_id, idx=target_voxel_idx, pick_ori=pick_ori)

    # Store results
    ve_dict = {}
    ve_dict["data"] = ve_data
    ve_dict["fs"] = evoked.info["sfreq"]
    ve_dict['peak_index'] = target_voxel_idx
    ve_dict['target_mni'] = mni_coord
    ve_dict['actual_mni'] = actual_pos
    ve_dict['subject_code'] = subject_code

    return ve_dict


def get_continuous_ve_target(mni_coord, meg_data, fwd, filters, chs_id, source_params=None, subject_code='fsaverage'):
    """
    Extract continuous virtual electrode data at a specific location for mTRF analysis.

    Parameters
    ----------
    mni_coord : array-like, shape (3,) or None
        MNI coordinates [x, y, z] in mm. If None, uses peak from STC.
    fwd : mne.Forward
        Forward model.
    meg_data : mne.io.Raw
        Continuous MEG data.
    filters : mne.beamformer.Beamformer
        Pre-computed beamformer filters.
    chs_id : str
        Channel type identifier.
    source_params : dict, optional
        Source space parameters (required for individual subjects).
    subject_code : str
        Subject code ('fsaverage' for template, subject_id for individual).

    Returns
    -------
    ve_dict : dict
        Dictionary containing continuous virtual electrode data.
    """
    src = fwd['src']

    # Use helper function to find target vertex
    target_voxel_idx, actual_pos = _find_target_vertex_index(mni_coord, src, subject_code, source_params)

    # Get beamformer weights directly to raw data for target voxel only
    weights = filters["weights"]

    # Get MEG data to apply weights to
    picks = functions_general.pick_chs(chs_id=chs_id, info=meg_data.info)
    meg_data_matrix = meg_data.get_data(picks=picks)  # Shape: (n_channels, n_times)

    # Combine orientations if needed
    if not source_params['source_estimation']:
        # Prepare array for 3 orientations
        base_idx = target_voxel_idx - (target_voxel_idx % 3)
        target_weights_x = weights[base_idx, :]
        target_weights_y = weights[base_idx + 1, :]
        target_weights_z = weights[base_idx + 2, :]

        # Apply beamformer weights directly to raw data for target voxel only
        ve_continuous = np.zeros((3, meg_data_matrix.shape[1]))
        ve_continuous[0, :] = target_weights_x.T @ meg_data_matrix
        ve_continuous[1, :] = target_weights_y.T @ meg_data_matrix
        ve_continuous[2, :] = target_weights_z.T @ meg_data_matrix

        comb = ve_continuous[0, :] ** 2
        comb += ve_continuous[1, :] ** 2
        comb += ve_continuous[2, :] ** 2
        comb = np.sqrt(comb)
        ve_continuous = comb

    else:
        # Get weights for target voxel
        target_weights = weights[target_voxel_idx, :]

        # Apply weights to continuous MEG data
        ve_continuous = target_weights.T @ meg_data_matrix  # Shape: (n_times,)

        # Reshape to match expected format
        ve_continuous = ve_continuous[np.newaxis, :]  # (1, n_times)

    # Store results
    ve_dict = {}
    ve_dict["data"] = ve_continuous#[np.newaxis, ...]  # (1, 1, n_times) to match epoch format
    ve_dict["fs"] = meg_data.info["sfreq"]
    ve_dict['peak_index'] = target_voxel_idx
    ve_dict['target_mni'] = mni_coord
    ve_dict['actual_mni'] = actual_pos
    ve_dict['subject_code'] = subject_code
    ve_dict['continuous'] = True  # Flag to indicate this is continuous data

    return ve_dict


def get_continuous_ve_peak(mni_coord, meg_data, fwd, filters, chs_id, source_params, subject_code='fsaverage'):
    """
    Extract continuous virtual electrode data at a specific location for mTRF analysis.

    Parameters
    ----------
    mni_coord : array-like, shape (3,) or None
        MNI coordinates [x, y, z] in mm. If None, uses peak from STC.
    fwd : mne.Forward
        Forward model.
    meg_data : mne.io.Raw
        Continuous MEG data.
    covs : mne.Covariance
        Covariance matrix.
    pick_ori : str | None
        Orientation picking method.
    subject_code : str
        Subject code ('fsaverage' for template, subject_id for individual).
    source_params : dict
        Source space parameters.

    Returns
    -------
    ve_dict : dict
        Dictionary containing continuous virtual electrode data.
    """
    import mne
    import os

    # Extract peak vertex index from MNI coordinates (no need to  morph because peak obtained in subject space)
    src = fwd['src']

    if src[0]['type'] == 'surf':
        lh_pos = src[0]['rr'][src[0]['vertno']] * 1000
        rh_pos = src[1]['rr'][src[1]['vertno']] * 1000
        all_pos = np.vstack([lh_pos, rh_pos])
    else:
        all_pos = src[0]['rr'][src[0]['vertno']] * 1000

    distances = np.linalg.norm(all_pos - mni_coord, axis=1)
    target_voxel_idx = np.argmin(distances)
    actual_pos = all_pos[target_voxel_idx]

    # Extract weights for target voxel
    weights = filters["weights"]

    # Get MEG data to apply weights to
    picks = functions_general.pick_chs(chs_id=chs_id, info=meg_data.info)
    meg_data_matrix = meg_data.get_data(picks=picks)  # Shape: (n_channels, n_times)

    # Combine orientations if needed
    if not source_params['source_estimation']:
        # Prepare array for 3 orientations
        base_idx = target_voxel_idx - (target_voxel_idx % 3)
        target_weights_x = weights[base_idx, :]
        target_weights_y = weights[base_idx + 1, :]
        target_weights_z = weights[base_idx + 2, :]

        # Apply beamformer weights directly to raw data for target voxel only
        ve_continuous = np.zeros((3, meg_data_matrix.shape[1]))
        ve_continuous[0, :] = target_weights_x.T @ meg_data_matrix
        ve_continuous[1, :] = target_weights_y.T @ meg_data_matrix
        ve_continuous[2, :] = target_weights_z.T @ meg_data_matrix

        comb = ve_continuous[0, :] ** 2
        comb += ve_continuous[1, :] ** 2
        comb += ve_continuous[2, :] ** 2
        comb = np.sqrt(comb)
        ve_continuous = comb

    else:
        # Get weights for target voxel
        target_weights = weights[target_voxel_idx, :]

        # Apply weights to continuous MEG data
        ve_continuous = target_weights.T @ meg_data_matrix  # Shape: (n_times,)

        # Reshape to match expected format
        ve_continuous = ve_continuous[np.newaxis, :]  # (1, n_times)

    # Store results
    ve_dict = {}
    ve_dict["data"] = ve_continuous  #[np.newaxis, ...]  # (1, 1, n_times) to match epoch format
    ve_dict["fs"] = meg_data.info["sfreq"]
    ve_dict['peak_index'] = target_voxel_idx
    ve_dict['target_mni'] = mni_coord
    ve_dict['actual_mni'] = actual_pos
    ve_dict['subject_code'] = subject_code
    ve_dict['continuous'] = True  # Flag to indicate this is continuous data

    return ve_dict
