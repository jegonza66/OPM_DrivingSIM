# -*- coding: utf-8 -*-
"""
TRF analysis on continuous source-space data at parcellation centroids.

Uses the parcellation source model from sourcemodel_setup.py to extract
continuous timecourses at label centroids (~150 for aparc.a2009s), then
fits mTRF models on that low-dimensional source representation.

Workflow:
1. Load MEG data and parcellation forward model
2. Compute/load LCMV beamformer filters
3. Apply beamformer to get continuous source timecourses at all centroids
4. Build/load event input arrays for TRF features
5. Fit mTRF on source data
6. Extract TRF coefficients as Evoked objects per feature
7. Grand average across subjects
"""

import os
import mne
import numpy as np
import mne.beamformer as beamformer
import functions_analysis
import functions_general
import load
import setup
import paths
import save
import plot_general
import matplotlib.pyplot as plt


# Load experiment info
exp_info = setup.exp_info()

# --------- Save / Display ---------#
use_saved_data = False
save_data = True
save_fig = True
display_figs = True
plot_individuals = True
if display_figs:
    plt.ion()
else:
    plt.ioff()

# --------- Parameters ---------#
meg_params = {'chs_id': 'mag_z',
              'band_id': None,
              'data_type': 'processed',
              'filter_sensors': True,
              }

# Source estimation parameters
parc = 'aparc'
spacing = 'ico4'  # Spacing used when creating the parcellation source model
pick_ori = None  # Must match sourcemodel_setup.py setting

# TRF parameters
trf_params = {
    'input_features': {
        # 'fix': None,
        # 'sac': None,
        # 'pur': None,
        'audio_env_std': None,
        # 'Steering_std_der': None,
        # 'left_but': None,
        # 'right_but': None,
    },
    'standarize': True,
    'fit_power': False,
    'alpha': None,
    'tmin': -0.2,
    'tmax': 0.5,
}
trf_params['baseline'] = (trf_params['tmin'], trf_params['tmax'])

# --------- Setup ---------#
subjects_dir = os.path.join(paths.mri_path, 'freesurfer')
os.environ["SUBJECTS_DIR"] = subjects_dir

# Build feature evokeds dictionary
feature_evokeds = {}
elements = trf_params['input_features'].keys()
for feature in elements:
    feature_evokeds[feature] = []
    if isinstance(trf_params['input_features'], dict):
        try:
            for value in trf_params['input_features'][feature]:
                feature_evokeds[f'{feature}-{value}'] = []
        except:
            pass

features = list(feature_evokeds.keys())

# Figure and save paths
fig_path = paths.plots_path + (f"TRF_Source_{meg_params['data_type']}/Band_{meg_params['band_id']}/parcellation_{parc}/"
                               f"{trf_params['input_features']}_{trf_params['tmin']}_{trf_params['tmax']}_"
                               f"bline{trf_params['baseline']}_alpha{trf_params['alpha']}_"
                               f"std{trf_params['standarize']}/{meg_params['chs_id']}/").replace(":", "")
save_path_trf = fig_path.replace(paths.plots_path, paths.save_path)

# --------- Run ---------#
for sub_idx, subject_id in enumerate(exp_info.subjects_ids):

    print(f"\n{'='*60}")
    print(f"Processing subject {subject_id}...")
    print(f"{'='*60}")

    subject = setup.subject(subject_id=subject_id)

    # --------- Determine subject code ---------#
    fs_subj_path = os.path.join(subjects_dir, subject_id)
    try:
        if len(os.listdir(fs_subj_path)):
            subject_code = subject_id
        else:
            subject_code = 'fsaverage'
    except:
        subject_code = 'fsaverage'

    # --------- Load MEG data ---------#
    meg_data = load.meg(subject_id=subject_id, meg_params=meg_params)
    picks = functions_general.pick_chs(chs_id=meg_params['chs_id'], info=meg_data.info)
    meg_data.pick(picks)
    meg_data.info.normalize_proj()

    # --------- Load parcellation forward model ---------#
    sources_path_subject = paths.sources_path + subject_id
    fname_fwd = sources_path_subject + f'/{subject_code}_{meg_params["data_type"]}_chs{meg_params["chs_id"]}_parcellation_{parc}-fwd.fif'
    fwd = mne.read_forward_solution(fname_fwd)
    # Restrict forward to channels present in MEG data (bad channels may have been dropped)
    fwd.pick_channels(meg_data.ch_names)
    src = fwd['src']

    # --------- Compute/Load LCMV beamformer ---------#
    fname_lcmv = (sources_path_subject +
                  f'/{subject_code}_{meg_params["data_type"]}_chs{meg_params["chs_id"]}'
                  f'_band{meg_params["band_id"]}_parcellation_{parc}_{pick_ori}-lcmv.h5')

    if os.path.isfile(fname_lcmv) and use_saved_data:
        filters = mne.beamformer.read_beamformer(fname_lcmv)
    else:
        data_cov = mne.compute_raw_covariance(meg_data)
        filters = beamformer.make_lcmv(info=meg_data.info, forward=fwd, data_cov=data_cov,
                                       reg=0.05, pick_ori=pick_ori)
        filters.save(fname=fname_lcmv, overwrite=True)

    # --------- Extract continuous source data at all centroids ---------#
    stc = beamformer.apply_lcmv_raw(meg_data, filters)
    source_data = stc.data  # (n_sources, n_times)
    n_sources = source_data.shape[0]

    # --------- Map source vertices to label names ---------#
    labels = mne.read_labels_from_annot(subject_code, parc=parc, subjects_dir=subjects_dir)

    lh_verts = src[0]['vertno']
    rh_verts = src[1]['vertno']

    label_names = []
    for vert in lh_verts:
        matched = False
        for label in labels:
            if label.hemi == 'lh' and vert in label.vertices:
                label_names.append(label.name)
                matched = True
                break
        if not matched:
            label_names.append(f'lh_vert{vert}')

    for vert in rh_verts:
        matched = False
        for label in labels:
            if label.hemi == 'rh' and vert in label.vertices:
                label_names.append(label.name)
                matched = True
                break
        if not matched:
            label_names.append(f'rh_vert{vert}')

    print(f'Extracted {n_sources} source timecourses ({len(lh_verts)} lh, {len(rh_verts)} rh)')

    # Create Raw object with source timecourses
    info_src = mne.create_info(label_names, meg_data.info['sfreq'], ch_types='misc')
    raw_src = mne.io.RawArray(source_data, info_src)

    # --------- Build input feature arrays ---------#
    # Use the original MEG data for feature construction (annotations, timing)
    subj_path = paths.save_path + f'TRF/{subject.subject_id}/'
    fname_bad_annot = 'bad_annot_array.pkl'

    if os.path.exists(subj_path + fname_bad_annot) and use_saved_data:
        bad_annotations_array = load.var(subj_path + fname_bad_annot)
        print('Loaded bad annotations array')
    else:
        print('Computing bad annotations array...')
        bad_annotations_array = functions_analysis.get_bad_annot_array(
            meg_data=meg_data, subj_path=subj_path, fname=fname_bad_annot)

    input_arrays = {}
    for feature in features:
        feature_data, fname_var = functions_analysis.load_input_array_feature(
            feature=feature, meg_params=meg_params, subj_path=subj_path, use_saved_data=use_saved_data)
        if isinstance(feature_data, np.ndarray):
            input_arrays[feature] = feature_data
        else:
            print(f'Computing input array for {feature}...')
            input_arrays = functions_analysis.make_mtrf_input(
                input_arrays=input_arrays, var_name=feature,
                subject=subject, meg_data=meg_data,
                bad_annotations_array=bad_annotations_array,
                subj_path=subj_path, fname=fname_var)

    model_input = np.array([input_arrays[key] for key in input_arrays.keys()]).T

    # --------- Fit TRF on source data ---------#
    trf_fname = f'TRF_{subject_id}.pkl'

    if os.path.isfile(save_path_trf + trf_fname) and use_saved_data:
        rf = load.var(save_path_trf + trf_fname)
        print('Loaded Source TRF')
    else:
        rf = functions_analysis.fit_mtrf(
            meg_data=raw_src, tmin=trf_params['tmin'], tmax=trf_params['tmax'],
            alpha=trf_params['alpha'] if trf_params['alpha'] else 0,
            model_input=model_input, chs_id='misc',
            standarize=trf_params['standarize'], fit_power=trf_params['fit_power'])

        if save_data:
            save.var(var=rf, path=save_path_trf, fname=trf_fname)

    # --------- Extract TRF coefficients as Evoked objects ---------#
    # rf.coef_ shape: (n_sources, n_features, n_delays)
    for feature_index, feature in enumerate(features):
        trf_coefs = rf.coef_[:, feature_index, :]  # (n_sources, n_delays)
        subj_evoked = mne.EvokedArray(data=trf_coefs, info=raw_src.info,
                                      tmin=trf_params['tmin'],
                                      baseline=trf_params['baseline'])

        feature_evokeds[feature].append(subj_evoked)

        if plot_individuals:
            fig = subj_evoked.plot(spatial_colors=False, gfp=True, show=display_figs,
                                  xlim=(trf_params['tmin'], trf_params['tmax']),
                                  titles=f'{feature} - {subject_id}')
            if save_fig:
                fig_path_subj = fig_path + f'{subject_id}/'
                fname_fig = f'{feature}_source_trf'
                save.fig(fig=fig, fname=fname_fig, path=fig_path_subj)

            # Brain plot
            stc_trf, src_full = functions_analysis.evoked_to_parcellation_stc(
                subj_evoked, parc, subject_code, subjects_dir, spacing)
            brain = plot_general.sources(
                stc=stc_trf, src=src_full, subject=subject_code,
                subjects_dir=subjects_dir, initial_time=None,
                surf_vol='surface', force_fsaverage=False,
                source_estimation='trf', views=['lateral', 'medial'],
                save_fig=save_fig, fig_path=fig_path + f'{subject_id}/',
                fname=f'{feature}_source_trf_brain')

# --------- Grand Average ---------#
print(f"\n{'='*60}")
print("Computing Grand Average...")
print(f"{'='*60}")

grand_avg = {}
for feature in features:
    grand_avg[feature] = mne.grand_average(feature_evokeds[feature], interpolate_bads=True)

    fig = grand_avg[feature].plot(spatial_colors=False, gfp=True, show=display_figs,
                                 xlim=(trf_params['tmin'], trf_params['tmax']),
                                 titles=f'GA - {feature}')
    if save_fig:
        fname_fig = f'{feature}_GA_source_trf'
        save.fig(fig=fig, fname=fname_fig, path=fig_path)

    # Grand average brain plot on fsaverage
    stc_ga, src_ga = functions_analysis.evoked_to_parcellation_stc(
        grand_avg[feature], parc, 'fsaverage', subjects_dir, spacing)
    brain = plot_general.sources(
        stc=stc_ga, src=src_ga, subject='fsaverage',
        subjects_dir=subjects_dir, initial_time=None,
        surf_vol='surface', force_fsaverage=True,
        source_estimation='trf', views=['lateral', 'medial'],
        save_fig=save_fig, fig_path=fig_path,
        fname=f'{feature}_GA_source_trf_brain')

# Save grand average
if save_data:
    save.var(var=grand_avg, path=save_path_trf, fname='GA_grand_avg.pkl')
    save.var(var=feature_evokeds, path=save_path_trf, fname='GA_all_subjects.pkl')

print(f"\nSource parcellation TRF analysis completed!")
print(f"Results saved to: {save_path_trf.split(paths.save_path)[-1]}")
