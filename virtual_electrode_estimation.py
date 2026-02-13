# -*- coding: utf-8 -*-
"""
Example script demonstrating how to use the sourcespace_ve.py module
for virtual electrode analysis.

This script shows how to:
1. Load preprocessed data and source space
2. Compute source reconstruction (STC)
3. Extract virtual electrode data at peak activation
4. Analyze evoked responses and Hilbert envelope
"""

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import sourcespace_ve
import load
import setup
import paths
import functions_analysis
import functions_general
import save
import plot_general
import pickle

# Setup experiment info and paths
exp_info = setup.exp_info()
save_path = paths.save_path
plot_path = paths.plots_path

#--------- Parameters ---------#
save_fig = True
save_data = True
display_figs = True
plot_individuals = True
use_saved_data = False

if display_figs:
    plt.ion()
else:
    plt.ioff()

# Task and trial parameters
task = 'DA'
trial_params = {
    'epoch_id': 'sac',  # epoch condition
    'reject': None,  # rejection criteria
    'evt_from_df': True
}

# Virtual electrode parameters
ve_params = {
    'mode_peak': 'pos',  # 'neg' for negative peak, 'pos' for positive peak 'abs' for absolute value
    'baseline_hilb': (-0.15, -0.05),  # baseline window for Hilbert envelope
    'use_mni_coordinate': True,  # Set to True to use specific MNI coordinate instead of peak
    'target_mni': [-28, -6, 64],  # MNI coordinates [x, y, z] in mm (example: left auditory cortex) F_MNI_coords = [[-28, -6, 64], [30, 0, 60]]
}

# mTRF parameters (only used if run_mtrf=True)
trf_params = {
    'run_mtrf': True,
    'input_features': ['sac'],   # Select features (events)
    'standarize': False,
    'alpha': None,
    'tmin': -0.2,
    'tmax': 0.5
}

# Source space parameters
source_params = {
    'source_estimation': 'evk',  # 'epo' / 'evk' / 'cov' / 'trf'
    'method': 'lcmv',
    'surf_vol': 'surface',
    'pick_ori': None,  # 'normal' / 'max-power' / 'vector'
    'ico': 4,  # Source space resolution
    'spacing': 10.  # Source space spacing
}

# MEG parameters
meg_params = {
    'chs_id': 'mag_z',
    'band_id': None,  # frequency band for filtering
    'filter_sensors': True,
    'filter_method': 'iir',
    'data_type': 'processed_annot'
}

# Define Subjects_dir as Freesurfer output folder
subjects_dir = os.path.join(paths.mri_path, 'freesurfer')
os.environ["SUBJECTS_DIR"] = subjects_dir

# Get frequency band
l_freq, h_freq = functions_general.get_freq_band(band_id=meg_params['band_id'])

# Define Grand average variables
feature_evokeds = {}
feature_evokeds_env = {}
for feature in trf_params['input_features']:
    feature_evokeds[feature] = []
    feature_evokeds_env[feature] = []

#--------- Analysis Loop ---------#
ve_results = []

for subject_id in exp_info.subjects_ids:  # Process first 3 subjects as example

    print(f"\nProcessing subject {subject_id}...")

    # Setup subject
    subject = setup.subject(subject_id=subject_id)

    if trf_params['run_mtrf']:
        trial_params['epoch_id'] = '+'.join(trf_params['input_features'])

    # Get times and baseline
    tmin, tmax, plot_xlim = functions_general.get_time_lims(
        epoch_id=trial_params['epoch_id'],
        subject=subject,
        plot_edge=0
    )
    baseline, plot_baseline = functions_general.get_baseline_duration(
        epoch_id=trial_params['epoch_id'],
        tmin=tmin,
        tmax=tmax
    )
    trf_params['baseline'] = baseline

    # Create coordinate identifier for filenames
    if ve_params['use_mni_coordinate']:
        coord_str = f"_{ve_params['target_mni'][0]}_{ve_params['target_mni'][1]}_{ve_params['target_mni'][2]}"
        method_str = "coord"
    else:
        coord_str = f"_{ve_params['mode_peak']}peak"
        method_str = "peak"

    # Create save paths
    # Main run path
    if trf_params['run_mtrf']:
        run_path = f'Band_{meg_params["band_id"]}/{trf_params['input_features']}_task{task}_{tmin}_{tmax}_bline{baseline}_{method_str}{coord_str}/'
    else:
        run_path = f'Band_{meg_params["band_id"]}/{trial_params['epoch_id']}_task{task}_{tmin}_{tmax}_bline{baseline}_{method_str}{coord_str}/'

    # Source model info path
    source_model_path = f'chs{meg_params['chs_id']}_{source_params['surf_vol']}_ico{source_params['ico']}_spacing{source_params['spacing']}_{source_params['pick_ori']}'
    sources_path_subject = paths.sources_path + subject.subject_id

    # LCMV filters path
    fname_lcmv = (f'/{subject_id}_{meg_params['data_type']}_band{meg_params['band_id']}_' + source_model_path + '-lcmv.h5')

    # TRF paths
    trf_save_path = save_path + f'TRF_VE_{meg_params["data_type"]}/' + run_path + f'std_{trf_params['standarize']}/' + source_model_path + '/'
    trf_fname = f'{subject_id}.pkl'
    ve_fig_path_trf = plot_path + f'TRF_VE_{meg_params["data_type"]}/' + run_path + f'std_{trf_params['standarize']}/' + source_model_path + '/'

    # VE paths
    source_model_path += f'_{source_params['source_estimation']}'
    ve_save_path = save_path + f'VE_{meg_params["data_type"]}/' + run_path + source_model_path + '/'
    ve_fig_path = plot_path + f'VE_{meg_params["data_type"]}/' + run_path + source_model_path + '/'
    ve_data_fname = f'Subject_{subject.subject_id}.pkl'

    # Check if subject has MRI data
    try:
        fs_subj_path = os.path.join(subjects_dir, subject.subject_id)
        os.listdir(fs_subj_path)
        subject_code = subject_id
    except:
        subject_code = 'fsaverage'

    if os.path.isfile(ve_save_path + ve_data_fname) and use_saved_data:
        # Try to load existing VE data
        with open(ve_save_path + ve_data_fname, 'rb') as f:
            ve_dict = pickle.load(f)
        print(f"Loaded existing VE data for subject {subject_id}")

    else:
        print(f"Computing VE data for subject {subject_id}...")

        # Load MEG data
        meg_data = load.meg(subject_id=subject_id, meg_params=meg_params)
        picks = functions_general.pick_chs(chs_id=meg_params['chs_id'], info=meg_data.info)
        meg_data.pick(picks)

        # Drop bad channels explicitly
        meg_data.drop_channels(meg_data.info['bads'])

        # Suppress warning about SSP projection
        meg_data.info.normalize_proj()

        # Epoch data
        epochs, events, onset_times = functions_analysis.epoch_data(
            subject=subject,
            epoch_id=trial_params['epoch_id'],
            meg_data=meg_data,
            tmin=tmin,
            tmax=tmax,
            from_df=trial_params['evt_from_df'],
            baseline=baseline
        )

        # Load forward model and compute covariance
        fwd = load.forward_model(sources_path_subject=sources_path_subject, subject_code=subject_code, chs_id=meg_params['chs_id'], source_params=source_params)

        # Load lcmv filter
        if os.path.isfile(sources_path_subject + fname_lcmv):
            filters = mne.beamformer.read_beamformer(sources_path_subject + fname_lcmv)
        else:
            data_cov = mne.compute_raw_covariance(meg_data)
            filters = mne.beamformer.make_lcmv(info=meg_data.info, forward=fwd, data_cov=data_cov, reg=0.05, pick_ori=source_params['pick_ori'])
            filters.save(fname=sources_path_subject + fname_lcmv, overwrite=True)

        # Extract virtual electrode data at specified location
        if ve_params['use_mni_coordinate']:
            if source_params['source_estimation'] == 'epo':
                # Use specific MNI coordinate
                ve_dict = sourcespace_ve.get_coordinate_ve_epochs(
                    mni_coord=ve_params['target_mni'],
                    fwd=fwd,
                    epochs=epochs,
                    filters=filters,  # Pass pre-computed filters
                    chs_id=meg_params['chs_id'],
                    pick_ori=source_params['pick_ori'],
                    subject_code=subject_code,
                    source_params=source_params
                )
            elif source_params['source_estimation'] == 'evk':
                # Use specific MNI coordinate
                ve_dict = sourcespace_ve.get_coordinate_ve_evoked(
                    mni_coord=ve_params['target_mni'],
                    fwd=fwd,
                    evoked=epochs.average(),
                    filters=filters,  # Pass pre-computed filters
                    chs_id=meg_params['chs_id'],
                    pick_ori=source_params['pick_ori'],
                    subject_code=subject_code,
                    source_params=source_params
                )
        else:
            if source_params['source_estimation'] == 'epo':

                stc = mne.beamformer.apply_lcmv(epochs.average(), filters)
                stc.apply_baseline(baseline=baseline)

                # Use automatic peak detection
                ve_dict = sourcespace_ve.get_peak_ve_epochs(
                    stc=stc,
                    fwd=fwd,
                    epochs=epochs,
                    filters=filters,  # Pass pre-computed filters
                    mode_peak=ve_params['mode_peak'],
                    pick_ori=source_params['pick_ori'],
                    subject_code=subject_code,
                    chs_id=meg_params['chs_id']
                )
            elif source_params['source_estimation'] == 'evk':

                stc = mne.beamformer.apply_lcmv(epochs.average(), filters)

                # Use automatic peak detection
                ve_dict = sourcespace_ve.get_peak_ve_evoked(
                    stc=stc,
                    fwd=fwd,
                    evoked=epochs.average(),
                    filters=filters,  # Pass pre-computed filters
                    mode_peak=ve_params['mode_peak'],
                    pick_ori=source_params['pick_ori'],
                    subject_code=subject_code,
                    chs_id=meg_params['chs_id']
                )

        # Apply baseline correction to VE data
        ve_dict['data'] = sourcespace_ve.apply_baseline_array(
            ve_data=ve_dict['data'],
            ve_times=epochs.times,
            baseline=baseline,
            source_estimation=source_params['source_estimation']
        )

        # Save VE data
        if save_data:
            os.makedirs(ve_save_path, exist_ok=True)
            save.var(var=ve_dict, path=ve_save_path, fname=ve_data_fname)

    if trf_params['run_mtrf']:
        print("Running mTRF analysis...")

        # Load MEG data and ensure same channel selection as filters
        meg_data_mtrf = load.meg(subject_id=subject_id, meg_params=meg_params)
        picks = functions_general.pick_chs(chs_id=meg_params['chs_id'], info=meg_data_mtrf.info)
        meg_data_mtrf.pick(picks)

        # Drop bad channels explicitly
        meg_data_mtrf.drop_channels(meg_data_mtrf.info['bads'])

        # Load forward model
        fwd = load.forward_model(sources_path_subject=sources_path_subject, subject_code=subject_code, chs_id=meg_params['chs_id'], source_params=source_params)

        # Compute STC using LCMV beamformer
        if os.path.isfile(sources_path_subject + fname_lcmv):
            filters = mne.beamformer.read_beamformer(sources_path_subject + fname_lcmv)
        else:
            data_cov = mne.compute_raw_covariance(meg_data_mtrf)
            filters = mne.beamformer.make_lcmv(info=meg_data_mtrf.info, forward=fwd, data_cov=data_cov, reg=0.05, pick_ori=source_params['pick_ori'])
            filters.save(fname=sources_path_subject + fname_lcmv, overwrite=True)

        # For mTRF, we need continuous VE data, not epoched
        if ve_params['use_mni_coordinate']:
            # Extract target voxel
            ve_continuous_dict = sourcespace_ve.get_continuous_ve_target(
                mni_coord=ve_params['target_mni'],
                filters=filters,
                fwd=fwd,
                meg_data=meg_data_mtrf,
                subject_code=subject_code,
                source_params=source_params,
                chs_id=meg_params['chs_id']
            )
        else:
            # Use peak location for continuous extraction
            ve_continuous_dict = sourcespace_ve.get_continuous_ve_peak(
                mni_coord=ve_dict['actual_mni'],
                filters=filters,
                fwd=fwd,
                meg_data=meg_data_mtrf,
                source_params=source_params,
                subject_code=subject_code,
                chs_id=meg_params['chs_id']
            )

        # Use existing functions_analysis.compute_trf on the continuous VE data
        # Create a temporary raw object with just the VE channel
        ve_data = ve_continuous_dict['data']#[:, 0, :]  # Extract (n_times,) from (1, 1, n_times)

        # Create minimal raw-like structure for TRF
        info_ve = mne.create_info(['VE'], meg_data_mtrf.info['sfreq'], ch_types='misc')
        raw_ve = mne.io.RawArray(ve_data, info_ve)
        raw_ve.set_annotations(meg_data_mtrf.annotations)  # Copy annotations for events

        # Create modified meg_params for VE
        ve_meg_params = meg_params.copy()
        ve_meg_params['chs_id'] = 'misc'  # VE is misc channel

        if os.path.isfile(trf_save_path + trf_fname) and use_saved_data:
            rf = load.var(trf_save_path + trf_fname)
            print('Loaded VE TRF')
        else:
            trf_params['fit_power'] = False  # Disable power fit

            rf = functions_analysis.compute_trf(
                subject=subject,
                meg_data=raw_ve,
                trf_params=trf_params,
                meg_params=ve_meg_params,
                from_df=trial_params['evt_from_df'],
                save_data=save_data,
                trf_path=trf_save_path,
                trf_fname=trf_fname
            )

        # Get TRF evoked responses
        evoked_ve, _ = functions_analysis.make_trf_evoked(
            subject=subject,
            rf=rf,
            meg_data=raw_ve,
            trf_params=trf_params,
            meg_params=ve_meg_params,
            evokeds=feature_evokeds,
            display_figs=False,
            plot_individuals=False,
            save_fig=False
        )

        # Repeat for power fit
        trf_save_path_aux = trf_save_path.replace(f'{trf_params["input_features"][0]}', f'power_{trf_params["input_features"][0]}')
        if os.path.isfile(trf_save_path_aux + trf_fname) and use_saved_data:
            rf = load.var(trf_save_path_aux + trf_fname)
            print('Loaded VE Power TRF')
        else:
            trf_params['fit_power'] = True  # Enable power fit

            rf = functions_analysis.compute_trf(
                subject=subject,
                meg_data=raw_ve,
                trf_params=trf_params,
                meg_params=ve_meg_params,
                from_df=trial_params['evt_from_df'],
                save_data=save_data,
                trf_path=trf_save_path_aux,
                trf_fname=trf_fname
            )

        # Get TRF evoked responses
        envelope, _ = functions_analysis.make_trf_evoked(
            subject=subject,
            rf=rf,
            meg_data=raw_ve,
            trf_params=trf_params,
            meg_params=ve_meg_params,
            evokeds=feature_evokeds_env,
            display_figs=False,
            plot_individuals=False,
            save_fig=False
        )

        # Make as arrays
        evoked_ve = {f'{key}': evoked_ve[key].data.squeeze() for key in evoked_ve.keys()}
        envelope = {f'{key}': envelope[key].data.squeeze() for key in envelope.keys()}

    else:

        if source_params['source_estimation'] == 'epo':
            evoked_ve = {f'{trial_params['epoch_id']}': ve_dict["data"].mean(axis=0).squeeze()}  # shape: (1, n_times,)
        elif source_params['source_estimation'] == 'evk':
            evoked_ve = {f'{trial_params['epoch_id']}': ve_dict["data"]}  # shape: (1, n_times)

        # Compute Hilbert envelope
        envelope = {f'{trial_params['epoch_id']}': sourcespace_ve.compute_hilbert_envelope(
            ve=ve_dict['data'],
            fs=ve_dict['fs'],
            tmin=tmin,
            baseline_hilb=ve_params['baseline_hilb'],
            source_estimation=source_params['source_estimation']
        )}

    # Store results
    subject_result = {
        'subject_id': subject_id,
        'peak_index': ve_dict['peak_index'],
        'evoked_ve': evoked_ve,
        'envelope': envelope,
        've_dict': ve_dict,
    }
    ve_results.append(subject_result)

    # Plot individual subject results - unified for both evoked and mTRF
    if plot_individuals:
        for key in evoked_ve.keys():

            plot_general.ve_evoked(evoked_ve=evoked_ve, envelope=envelope, key=key, trf_params=trf_params,
                                   ve_params=ve_params, ve_dict=ve_dict, tmin=tmin, tmax=tmax, subject_id=subject_id,
                                   save_fig=save_fig, ve_fig_path_trf=ve_fig_path_trf, ve_fig_path=ve_fig_path)

#--------- Group Analysis ---------#
print("\nPerforming group analysis...")

# Combine evoked responses
all_evoked_data = {}
all_envelopes = {}

for result in ve_results:
    for key in result['evoked_ve'].keys():
        if key not in all_evoked_data:
            all_evoked_data[key] = []
            all_envelopes[key] = []
        all_evoked_data[key].append(result['evoked_ve'][key])
        all_envelopes[key].append(result['envelope'][key])

# Iterate over features (only usefull in mtrf)
for key in all_evoked_data.keys():
    # Convert to arrays
    all_evoked_data_arr = np.array(all_evoked_data[key])
    all_envelopes_arr = np.array(all_envelopes[key])

    # Compute group averages
    grand_avg_evoked = np.mean(all_evoked_data_arr, axis=0)
    grand_avg_envelope = np.mean(all_envelopes_arr, axis=0)

    # Plot group results
    if display_figs:
        plot_general.ve_ga(ve_results=ve_results, grand_avg_evoked=grand_avg_evoked, grand_avg_envelope=grand_avg_envelope,
                           key=key, all_evoked_data_arr=all_evoked_data_arr, all_envelopes_arr=all_envelopes_arr, trf_params=trf_params,
                           tmin=tmin, tmax=tmax, save_fig=save_fig, ve_fig_path_trf=ve_fig_path_trf, ve_fig_path=ve_fig_path)

    # Save group results
    if save_data:

        group_results = {
            'grand_avg_evoked': grand_avg_evoked,
            'grand_avg_envelope': grand_avg_envelope,
            'all_evoked_data': all_evoked_data_arr,
            'all_envelopes': all_envelopes_arr,
            'subject_ids': [r['subject_id'] for r in ve_results],
            'peak_indices': [r['peak_index'] for r in ve_results],
            've_params': ve_params,
            'meg_params': meg_params
        }

        group_fname = f'GA_{key}.pkl'
        if trf_params['run_mtrf']:
            save.var(var=group_results, path=trf_save_path, fname=group_fname)
        else:
            save.var(var=group_results, path=ve_save_path, fname=group_fname)

print("Virtual electrode analysis completed!")
if trf_params['run_mtrf']:
    print(f"Results saved to: {trf_save_path.split(paths.save_path)[-1]}")
else:
    print(f"Results saved to: {ve_save_path.split(paths.save_path)[-1]}")
