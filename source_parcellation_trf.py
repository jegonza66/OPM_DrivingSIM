# -*- coding: utf-8 -*-
"""
TRF analysis on continuous source-space data at parcellation centroids.

Uses the parcellation source model from sourcemodel_setup.py to extract
continuous timecourses at label centroids, then fits mTRF models on that
low-dimensional source representation.

Supports both surface parcellations (aparc, aparc.a2009s) and volume
parcellations (AAL, Harvard-Oxford, Schaefer, or custom NIfTI atlases).

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
import nibabel as nib
import mne.beamformer as beamformer
import functions_analysis
import functions_general
import load
import setup
import paths
import save
import plot_general
import matplotlib.pyplot as plt
from mne.transforms import apply_trans, read_ras_mni_t


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
surf_vol = 'vol_parcellation'  # 'parcellation' | 'vol_parcellation'
parc = 'aparc.a2009s'          # Surface parcellation (used when surf_vol='parcellation')
# Volume atlas (used when surf_vol='vol_parcellation') — must match sourcemodel_setup.py
vol_parc_atlas = 'aal'         # 'aal' | 'destrieux' | 'harvard_oxford' | 'schaefer' | or path to .nii.gz
pick_ori = None  # Must match sourcemodel_setup.py setting
spacing = 'ico4'  # Must match sourcemodel_setup.py setting
pos = 10          # Must match sourcemodel_setup.py setting (volume grid spacing in mm)

# TRF parameters
trf_params = {
    'input_features': {
        'fix': None,
        'sac': None,
        'pur': None,
        'audio_env_std': None,
        'Steering_std_der': None,
        'left_but': None,
        'right_but': None,
    },
    'standarize': True,
    'fit_power': False,
    'alpha': None,
    'tmin': -0.2,
    'tmax': 0.5,
}
trf_params['baseline'] = (trf_params['tmin'], trf_params['tmax'])

initial_time = None

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
if surf_vol == 'vol_parcellation':
    # Resolve atlas name for file paths
    if os.path.isfile(vol_parc_atlas):
        vol_parc_name = os.path.basename(vol_parc_atlas).replace('.nii.gz', '').replace('.nii', '')
    else:
        vol_parc_name = vol_parc_atlas
    parc_tag = f'vol_parcellation_{vol_parc_name}'
else:
    parc_tag = f'parcellation_{parc}'

fig_path = paths.plots_path + (f"TRF_Source_{meg_params['data_type']}/Band_{meg_params['band_id']}/{parc_tag}/"
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
    fname_fwd = sources_path_subject + f'/{subject_code}_{meg_params["data_type"]}_chs{meg_params["chs_id"]}_{parc_tag}-fwd.fif'
    fwd = mne.read_forward_solution(fname_fwd)
    # Restrict forward to channels present in MEG data (bad channels may have been dropped)
    fwd.pick_channels(meg_data.ch_names)
    src = fwd['src']

    # --------- Compute/Load LCMV beamformer ---------#
    fname_lcmv = (sources_path_subject +
                  f'/{subject_code}_{meg_params["data_type"]}_chs{meg_params["chs_id"]}'
                  f'_band{meg_params["band_id"]}_{parc_tag}_{pick_ori}-lcmv.h5')

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
    if surf_vol == 'parcellation':
        # Surface parcellation: two hemispheres, use read_labels_from_annot
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

    elif surf_vol == 'vol_parcellation':
        # Volume parcellation: single source space, look up atlas labels
        # by transforming centroid positions to MNI152 and reading the atlas
        if os.path.isfile(vol_parc_atlas):
            atlas_img = nib.load(vol_parc_atlas)
            # Try to load a companion labels file, otherwise use integer IDs
            vol_label_names_map = None
        else:
            from nilearn import datasets as ni_datasets
            if vol_parc_atlas == 'aal':
                atlas = ni_datasets.fetch_atlas_aal()
            elif vol_parc_atlas == 'destrieux':
                atlas = ni_datasets.fetch_atlas_destrieux_2009()
            elif vol_parc_atlas == 'harvard_oxford':
                atlas = ni_datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
            elif vol_parc_atlas == 'schaefer':
                atlas = ni_datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
            else:
                raise ValueError(f'Unknown vol_parc_atlas: {vol_parc_atlas}')
            atlas_img = nib.load(atlas['maps'])
            # Build mapping: integer ID → label name
            atlas_data_full = np.asarray(atlas_img.dataobj).astype(int)
            atlas_label_list = atlas.get('labels', None)
            if atlas_label_list is not None:
                unique_ids = sorted(np.unique(atlas_data_full[atlas_data_full != 0]))
                # AAL labels list already matches the integer order
                if len(atlas_label_list) == len(unique_ids):
                    vol_label_names_map = {uid: str(atlas_label_list[i])
                                           for i, uid in enumerate(unique_ids)}
                else:
                    # labels list includes background as first entry
                    vol_label_names_map = {}
                    for i, uid in enumerate(unique_ids):
                        idx = uid if uid < len(atlas_label_list) else i
                        vol_label_names_map[uid] = str(atlas_label_list[idx])
            else:
                vol_label_names_map = None

        atlas_data = np.asarray(atlas_img.dataobj).astype(int)
        atlas_affine = atlas_img.affine
        inv_atlas_affine = np.linalg.inv(atlas_affine)

        # MNI305 → MNI152 (fixed coordinate-system relationship)
        # Published by FreeSurfer:
        # https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems
        mni305_to_mni152 = np.array([
            [ 0.9975, -0.0073,  0.0176, -0.0429],
            [ 0.0146,  1.0009, -0.0024,  1.5496],
            [-0.0130, -0.0093,  0.9971,  1.1840],
            [ 0.0000,  0.0000,  0.0000,  1.0000],
        ])

        # Transform centroid vertex positions to MNI152 to find their parcel ID
        verts = src[0]['vertno']
        vert_rr_mm = src[0]['rr'][verts] * 1000  # m → mm (surface-RAS)

        ras_mni_t = read_ras_mni_t(subject_code, subjects_dir)
        vert_mni305_mm = apply_trans(ras_mni_t, vert_rr_mm)
        vert_mni152_mm = apply_trans(mni305_to_mni152, vert_mni305_mm)

        vert_vox = apply_trans(inv_atlas_affine, vert_mni152_mm)
        vert_vox_idx = np.round(vert_vox).astype(int)
        for dim in range(3):
            vert_vox_idx[:, dim] = np.clip(vert_vox_idx[:, dim], 0, atlas_data.shape[dim] - 1)

        vert_parcel_ids = atlas_data[vert_vox_idx[:, 0], vert_vox_idx[:, 1], vert_vox_idx[:, 2]]

        label_names = []
        for p_id in vert_parcel_ids:
            if vol_label_names_map is not None and p_id in vol_label_names_map:
                label_names.append(vol_label_names_map[p_id])
            else:
                label_names.append(f'parcel_{p_id}')

    print(f'Extracted {n_sources} source timecourses ({len(label_names)} labels)')

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

            # Brain plot (surface parcellations only)
            if surf_vol == 'parcellation':
                stc_trf, src_full = functions_analysis.evoked_to_parcellation_stc(
                    subj_evoked, parc, subject_code, subjects_dir, spacing)
                brain = plot_general.sources(
                    stc=stc_trf, src=src_full, subject=subject_code,
                    subjects_dir=subjects_dir, initial_time=initial_time,
                    surf_vol='surface', force_fsaverage=False,
                    source_estimation='trf', views=['lateral', 'medial'],
                    save_fig=save_fig, fig_path=fig_path + f'{subject_id}/',
                    fname=f'{feature}_source_trf_brain')

            elif surf_vol == 'vol_parcellation':
                # Create VolSourceEstimate from TRF coefficients
                # Fill ALL vertices in each parcel with the centroid's value
                # for uniform region coloring (not just sparse centroid points)
                fname_src_vol = (paths.sources_path + subject_code
                                 + f'/{subject_code}_volume_{spacing}_{int(pos)}-src.fif')
                src_vol_full = mne.read_source_spaces(fname_src_vol)

                # Map full volume vertices to MNI152 atlas parcels
                full_inuse = src_vol_full[0]['inuse'].astype(bool)
                full_rr_mm = src_vol_full[0]['rr'][full_inuse] * 1000

                ras_mni_t_plot = read_ras_mni_t(subject_code, subjects_dir)
                full_mni305 = apply_trans(ras_mni_t_plot, full_rr_mm)
                full_mni152 = apply_trans(mni305_to_mni152, full_mni305)
                full_vox = apply_trans(inv_atlas_affine, full_mni152)
                full_vox_idx = np.round(full_vox).astype(int)
                for dim in range(3):
                    full_vox_idx[:, dim] = np.clip(full_vox_idx[:, dim], 0, atlas_data.shape[dim] - 1)
                full_parcel_ids = atlas_data[full_vox_idx[:, 0], full_vox_idx[:, 1], full_vox_idx[:, 2]]

                # Build mapping: parcel ID → TRF centroid index
                parc_vertno = src[0]['vertno']
                parcel_to_trf = {}
                for i, v in enumerate(parc_vertno):
                    v_inuse_idx = np.searchsorted(src_vol_full[0]['vertno'], v)
                    if v_inuse_idx < len(full_parcel_ids):
                        p_id = full_parcel_ids[v_inuse_idx]
                        parcel_to_trf[p_id] = i

                # Fill all vertices with their parcel's TRF value
                full_vertno = src_vol_full[0]['vertno']
                full_data = np.zeros((len(full_vertno), trf_coefs.shape[1]))
                for v_idx, p_id in enumerate(full_parcel_ids):
                    if p_id in parcel_to_trf:
                        full_data[v_idx] = trf_coefs[parcel_to_trf[p_id]]

                stc_vol = mne.VolSourceEstimate(
                    data=full_data,
                    vertices=[full_vertno],
                    tmin=trf_params['tmin'],
                    tstep=1 / raw_src.info['sfreq']
                )
                brain = plot_general.sources(
                    stc=stc_vol, src=src_vol_full, subject=subject_code,
                    subjects_dir=subjects_dir, initial_time=initial_time,
                    surf_vol='volume', force_fsaverage=False,
                    source_estimation='trf', views=['lateral', 'medial'],
                    alpha=0.5,
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

    # Grand average brain plot on fsaverage (surface parcellations only)
    if surf_vol == 'parcellation':
        stc_ga, src_ga = functions_analysis.evoked_to_parcellation_stc(
            grand_avg[feature], parc, 'fsaverage', subjects_dir, spacing)
        brain = plot_general.sources(
            stc=stc_ga, src=src_ga, subject='fsaverage',
            subjects_dir=subjects_dir, initial_time=initial_time,
            surf_vol='surface', force_fsaverage=True,
            source_estimation='trf', views=['lateral', 'medial'],
            save_fig=save_fig, fig_path=fig_path,
            fname=f'{feature}_GA_source_trf_brain')

    elif surf_vol == 'vol_parcellation':
        # Grand average volume brain plot on fsaverage
        # Fill all vertices in each parcel with that parcel's TRF value
        fname_src_fsavg = (paths.sources_path + 'fsaverage'
                           + f'/fsaverage_volume_{spacing}_{int(pos)}-src.fif')
        src_fsavg = mne.read_source_spaces(fname_src_fsavg)

        fname_src_parc_fsavg = (paths.sources_path + 'fsaverage'
                                + f'/fsaverage_vol_parcellation_{vol_parc_name}-src.fif')
        src_parc_fsavg = mne.read_source_spaces(fname_src_parc_fsavg)

        # Map fsaverage volume vertices to atlas parcels
        fsavg_inuse = src_fsavg[0]['inuse'].astype(bool)
        fsavg_rr_mm = src_fsavg[0]['rr'][fsavg_inuse] * 1000
        ras_mni_t_ga = read_ras_mni_t('fsaverage', subjects_dir)
        fsavg_mni305 = apply_trans(ras_mni_t_ga, fsavg_rr_mm)
        fsavg_mni152 = apply_trans(mni305_to_mni152, fsavg_mni305)
        fsavg_vox = apply_trans(inv_atlas_affine, fsavg_mni152)
        fsavg_vox_idx = np.round(fsavg_vox).astype(int)
        for dim in range(3):
            fsavg_vox_idx[:, dim] = np.clip(fsavg_vox_idx[:, dim], 0, atlas_data.shape[dim] - 1)
        fsavg_parcel_ids = atlas_data[fsavg_vox_idx[:, 0], fsavg_vox_idx[:, 1], fsavg_vox_idx[:, 2]]

        # Build mapping: parcel ID → TRF centroid index
        parc_vertno_ga = src_parc_fsavg[0]['vertno']
        parcel_to_trf_ga = {}
        for i, v in enumerate(parc_vertno_ga):
            v_idx = np.searchsorted(src_fsavg[0]['vertno'], v)
            if v_idx < len(fsavg_parcel_ids):
                p_id = fsavg_parcel_ids[v_idx]
                parcel_to_trf_ga[p_id] = i

        # Fill all vertices with their parcel's GA TRF value
        full_vertno_ga = src_fsavg[0]['vertno']
        full_data_ga = np.zeros((len(full_vertno_ga), grand_avg[feature].data.shape[1]))
        for v_idx, p_id in enumerate(fsavg_parcel_ids):
            if p_id in parcel_to_trf_ga:
                full_data_ga[v_idx] = grand_avg[feature].data[parcel_to_trf_ga[p_id]]

        stc_ga_vol = mne.VolSourceEstimate(
            data=full_data_ga,
            vertices=[full_vertno_ga],
            tmin=grand_avg[feature].times[0],
            tstep=1 / grand_avg[feature].info['sfreq']
        )
        brain = plot_general.sources(
            stc=stc_ga_vol, src=src_fsavg, subject='fsaverage',
            subjects_dir=subjects_dir, initial_time=initial_time,
            surf_vol='volume', force_fsaverage=True,
            source_estimation='trf', views=['lateral', 'medial'],
            alpha=0.5,
            save_fig=save_fig, fig_path=fig_path,
            fname=f'{feature}_GA_source_trf_brain')

# Save grand average
if save_data:
    save.var(var=grand_avg, path=save_path_trf, fname='GA_grand_avg.pkl')
    save.var(var=feature_evokeds, path=save_path_trf, fname='GA_all_subjects.pkl')

print(f"\nSource parcellation TRF analysis completed!")
print(f"Results saved to: {save_path_trf.split(paths.save_path)[-1]}")
