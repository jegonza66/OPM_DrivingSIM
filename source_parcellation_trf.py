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
use_saved_data = True
save_data = True
save_fig = True
display_figs = True
plot_individuals = True
if display_figs:
    plt.ion()
else:
    plt.ioff()

# --------- Statistics (cluster-based permutations across regions) ---------#
run_permutations = True
pval_threshold = 0.05               # significance level for clusters
t_thresh = dict(start=0, step=0.2)  # TFCE; or a float for a fixed t-threshold
n_permutations = 1024
plot_significance = True             # masked GA brain plots of significant regions

# --------- Parameters ---------#
meg_params = {'chs_id': 'mag_z',
              'band_id': None,
              'data_type': 'processed',
              'filter_sensors': True,
              }

# Source estimation parameters
surf_vol = 'parcellation'  # 'parcellation' | 'vol_parcellation'
parc = 'aparc.a2009s'          # Surface parcellation (used when surf_vol='parcellation')
# Volume atlas (used when surf_vol='vol_parcellation') — must match sourcemodel_setup.py
vol_parc_atlas = 'aal'         # 'aal' | 'destrieux' | 'harvard_oxford' | 'schaefer' | or path to .nii.gz
pick_ori = None  # Must match sourcemodel_setup.py setting
spacing = 'ico4'  # Must match sourcemodel_setup.py setting
pos = 10          # Must match sourcemodel_setup.py setting (volume grid spacing in mm)

# TRF parameters
trf_params = {
    'input_features': {
        'fix': None,#['CF','DA', 'Audio'],
        'sac': None,#['CF','DA', 'Audio'],
        # 'pur': ['CF','DA', 'Audio'],
        'audio_env_std': None,
        'Steering_std_der': None,
        'Gas_std_der': None,
        'Brake_std_der': None,
        'left_but': None,
        'right_but': None,
    },
    'standarize': True,
    'fit_power': False,
    'alpha': [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000],
    # Alpha cross-validation: k-fold over contiguous temporal blocks.
    # cv_aggregate: 'mean_fisher' (default, Fisher-z averaged per-fold correlation),
    #               'mean' (plain average), or 'pool' (one correlation over pooled preds)
    'cv_n_splits': 5,
    'cv_aggregate': 'mean_fisher',
    # Per-feature duration: use dict with 'default' key and optional per-feature overrides
    # e.g. 'tmin': {'default': -0.2, 'left_but': -2}, 'tmax': {'default': 0.5, 'left_but': 2}
    'tmin': {'default': -0.2, 'Steering_std_der': -2, 'left_but': -2, 'right_but': -2, 'Gas_std_der': -2, 'Brake_std_der': -2},
    'tmax': {'default': 0.5, 'Steering_std_der': 2, 'left_but': 2, 'right_but': 2, 'Gas_std_der': 2, 'Brake_std_der': 2},
    'plot_margin': 0.15,  # seconds to crop from each side of plotted TRF time series
}

# Per-feature initial time for brain plots: dict with 'default' key and optional per-feature overrides
# e.g. {'default': None, 'fix': 0.1, 'left_but': 0.3}
initial_time = {'default': None,
                'fix': None,
                'sac': [0.0, 0.12],
                'pur': None,
                'audio_env_std': 0.0,
                'Steering_std_der': 0.0,
                'Gas_std_der': 0.0,
                'Brake_std_der': 0.0,
                'left_but': 0.0,
                'right_but': 0.0,
                }

# --------- Setup ---------#
subjects_dir = os.path.join(paths.mri_path, 'freesurfer')
os.environ["SUBJECTS_DIR"] = subjects_dir

# Build feature evokeds dictionary
feature_evokeds = {}
features = functions_analysis.expand_features(trf_params['input_features'])
for feature in features:
    feature_evokeds[feature] = []

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

_path_tmin = trf_params['tmin'].get('default') if isinstance(trf_params['tmin'], dict) else trf_params['tmin']
_path_tmax = trf_params['tmax'].get('default') if isinstance(trf_params['tmax'], dict) else trf_params['tmax']
_path_bline = (_path_tmin, _path_tmax)
_alpha_tag = f'_alpha{trf_params["alpha"]}' if trf_params['alpha'] is None else '_alphaCV'
_trf_prefix = 'TRF_ENV_Source' if trf_params['fit_power'] else 'TRF_Source'
fig_path = paths.plots_path + (f"{_trf_prefix}_{meg_params['data_type']}/Band_{meg_params['band_id']}/{parc_tag}/"
                               f"{functions_general.features_path_str(trf_params['input_features'])}_{_path_tmin}_{_path_tmax}_"
                               f"bline{_path_bline}{_alpha_tag}_"
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
        for i, p_id in enumerate(vert_parcel_ids):
            if vol_label_names_map is not None and p_id in vol_label_names_map:
                label_names.append(vol_label_names_map[p_id])
            else:
                # Give each background centroid a unique name to avoid
                # MNE's duplicate-renaming (which adds '-0', '-1' suffixes
                # that break name-based matching later)
                label_names.append(f'parcel_{p_id}_src{i}')

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

        # --------- Fit TRF on source data (per-duration groups) ---------#
    trf_fname = f'TRF_{subject_id}.pkl'
    duration_groups = functions_analysis.group_features_by_duration(features, trf_params)
    plot_margin = trf_params.get('plot_margin', 0)

    if os.path.isfile(save_path_trf + trf_fname) and use_saved_data:
        rf_results = load.var(save_path_trf + trf_fname)
        # Handle legacy saved format (single RF object)
        if not isinstance(rf_results, list):
            default_tmin, default_tmax = functions_analysis.get_feature_tmin_tmax(features[0], trf_params)
            rf_results = [{'rf': rf_results, 'features': features, 'tmin': default_tmin, 'tmax': default_tmax}]
        print('Loaded Source TRF')
    else:
        alpha = trf_params['alpha'] if trf_params['alpha'] else 0
        rf_results = []
        for (group_tmin, group_tmax), group_features in duration_groups.items():
            group_input = np.array([input_arrays[f] for f in group_features]).T
            print(f'Fitting mTRF (tmin={group_tmin}, tmax={group_tmax}) for features: {group_features}')
            group_rf = functions_analysis.fit_mtrf(
                meg_data=raw_src, tmin=group_tmin, tmax=group_tmax,
                alpha=alpha,
                model_input=group_input, chs_id='misc',
                n_splits=trf_params.get('cv_n_splits', 5),
                cv_aggregate=trf_params.get('cv_aggregate', 'mean_fisher'),
                standarize=trf_params['standarize'], fit_power=trf_params['fit_power'])
            best_alpha = functions_analysis.extract_best_alpha(group_rf)
            if best_alpha is not None:
                print(f'  Best alpha: {best_alpha}')
            rf_results.append({
                'rf': group_rf,
                'features': group_features,
                'tmin': group_tmin,
                'tmax': group_tmax,
                'best_alpha': best_alpha,
            })

        # Save alpha report
        functions_analysis.save_alpha_report(rf_results, subject_id, fig_path + f'{subject_id}/')

        if save_data:
            save.var(var=rf_results, path=save_path_trf, fname=trf_fname)

    # --------- Compute label positions for spatial coloring ---------#
    label_positions = {}
    if surf_vol == 'parcellation':
        lh_verts_pos = src[0]['rr'][src[0]['vertno']]
        rh_verts_pos = src[1]['rr'][src[1]['vertno']]
        all_pos = np.vstack([lh_verts_pos, rh_verts_pos])
        for idx_lp, name_lp in enumerate(label_names):
            label_positions[name_lp] = all_pos[idx_lp]
    elif surf_vol == 'vol_parcellation':
        vol_verts_pos = src[0]['rr'][src[0]['vertno']]
        for idx_lp, name_lp in enumerate(label_names):
            label_positions[name_lp] = vol_verts_pos[idx_lp]

    # --------- Extract TRF coefficients as Evoked objects ---------#
    for group in rf_results:
        for feat_idx, feature in enumerate(group['features']):
            feat_tmin = group['tmin']
            feat_tmax = group['tmax']
            trf_coefs = group['rf'].coef_[:, feat_idx, :]  # (n_sources, n_delays)
            subj_evoked = mne.EvokedArray(data=trf_coefs, info=raw_src.info,
                                           tmin=feat_tmin,
                                           baseline=(feat_tmin, feat_tmax))

            feature_evokeds[feature].append(subj_evoked)

            feat_initial_times = functions_analysis.get_feature_initial_time(feature, initial_time)

            if plot_individuals:
                # Spatial-colored evoked plot with head diagram
                fig = plot_general.plot_source_evoked_spatial(
                    evoked=subj_evoked, label_positions=label_positions, gfp=True,
                    xlim=(feat_tmin + plot_margin, feat_tmax - plot_margin),
                    title=f'{feature} - {subject_id}',
                    display_figs=display_figs, save_fig=save_fig,
                    fig_path=fig_path + f'{subject_id}/',
                    fname=f'{feature}_source_trf')

                # Brain plot (surface parcellations only)
                if surf_vol == 'parcellation':
                    stc_trf, src_full = functions_analysis.evoked_to_parcellation_stc(
                        subj_evoked, parc, subject_code, subjects_dir, spacing)
                    for it in feat_initial_times:
                        it_suffix = f'_{it}s' if it is not None else ''
                        brain = plot_general.sources(
                            stc=stc_trf, src=src_full, subject=subject_code,
                            subjects_dir=subjects_dir, initial_time=it,
                            surf_vol='surface', force_fsaverage=False,
                            source_estimation='trf', views=['lateral', 'medial'],
                            plot_margin=plot_margin,
                            save_fig=save_fig, fig_path=fig_path + f'{subject_id}/',
                            fname=f'{feature}_source_trf_brain{it_suffix}')

                elif surf_vol == 'vol_parcellation':
                    # Create VolSourceEstimate from TRF coefficients
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

                    # Build mapping: parcel ID -> TRF centroid index
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
                        tmin=feat_tmin,
                        tstep=1 / raw_src.info['sfreq']
                    )
                    for it in feat_initial_times:
                        it_suffix = f'_{it}s' if it is not None else ''
                        brain = plot_general.sources(
                            stc=stc_vol, src=src_vol_full, subject=subject_code,
                            subjects_dir=subjects_dir, initial_time=it,
                            surf_vol='volume', force_fsaverage=False,
                            source_estimation='trf', views=['lateral', 'medial'],
                            alpha=0.5, plot_margin=plot_margin,
                            save_fig=save_fig, fig_path=fig_path + f'{subject_id}/',
                            fname=f'{feature}_source_trf_brain{it_suffix}')


# --------- Grand Average ---------#
print(f"\n{'='*60}")
print("Computing Grand Average...")
print(f"{'='*60}")

# Compute label positions for GA spatial coloring (using fsaverage for surface)
ga_label_positions = {}
if surf_vol == 'parcellation':
    ga_labels = mne.read_labels_from_annot('fsaverage', parc=parc, subjects_dir=subjects_dir)
    fname_src_ga = paths.sources_path + f'fsaverage/fsaverage_surface_{spacing}-src.fif'
    if os.path.isfile(fname_src_ga):
        src_ga_space = mne.read_source_spaces(fname_src_ga)
        lh_verts_ga = src_ga_space[0]['vertno']
        rh_verts_ga = src_ga_space[1]['vertno']
        lh_pos_ga = src_ga_space[0]['rr'][lh_verts_ga]
        rh_pos_ga = src_ga_space[1]['rr'][rh_verts_ga]
        all_pos_ga = np.vstack([lh_pos_ga, rh_pos_ga])
        # Build label names for fsaverage (same parcellation)
        ga_label_names_tmp = []
        for vert in lh_verts_ga:
            matched = False
            for label in ga_labels:
                if label.hemi == 'lh' and vert in label.vertices:
                    ga_label_names_tmp.append(label.name)
                    matched = True
                    break
            if not matched:
                ga_label_names_tmp.append(f'lh_vert{vert}')
        for vert in rh_verts_ga:
            matched = False
            for label in ga_labels:
                if label.hemi == 'rh' and vert in label.vertices:
                    ga_label_names_tmp.append(label.name)
                    matched = True
                    break
            if not matched:
                ga_label_names_tmp.append(f'rh_vert{vert}')
        for idx_lp, name_lp in enumerate(ga_label_names_tmp):
            ga_label_positions[name_lp] = all_pos_ga[idx_lp]

plot_margin = trf_params.get('plot_margin', 0)

grand_avg = {}
for feature in features:
    grand_avg[feature] = mne.grand_average(feature_evokeds[feature], interpolate_bads=True)

    feat_tmin, feat_tmax = functions_analysis.get_feature_tmin_tmax(feature, trf_params)
    feat_initial_times = functions_analysis.get_feature_initial_time(feature, initial_time)

    # Spatial-colored evoked plot with head diagram
    fig = plot_general.plot_source_evoked_spatial(
        evoked=grand_avg[feature], label_positions=ga_label_positions if ga_label_positions else label_positions,
        gfp=True, xlim=(feat_tmin + plot_margin, feat_tmax - plot_margin),
        title=f'GA - {feature}',
        display_figs=display_figs, save_fig=save_fig,
        fig_path=fig_path, fname=f'{feature}_GA_source_trf')

    # Grand average brain plot on fsaverage (surface parcellations only)
    if surf_vol == 'parcellation':
        stc_ga, src_ga = functions_analysis.evoked_to_parcellation_stc(
            grand_avg[feature], parc, 'fsaverage', subjects_dir, spacing)
        for it in feat_initial_times:
            it_suffix = f'_{it}s' if it is not None else ''
            brain = plot_general.sources(
                stc=stc_ga, src=src_ga, subject='fsaverage',
                subjects_dir=subjects_dir, initial_time=it,
                surf_vol='surface', force_fsaverage=True,
                source_estimation='trf', views=['lateral', 'medial'],
                plot_margin=plot_margin,
                save_fig=save_fig, fig_path=fig_path,
                fname=f'{feature}_GA_source_trf_brain{it_suffix}')

    elif surf_vol == 'vol_parcellation':
        # Grand average volume brain plot on fsaverage
        # Fill all vertices in each parcel with that parcel's TRF value
        fname_src_fsavg = (paths.sources_path + 'fsaverage'
                           + f'/fsaverage_volume_{spacing}_{int(pos)}-src.fif')
        src_fsavg = mne.read_source_spaces(fname_src_fsavg)

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

        # Build mapping: parcel ID → label name (for matching to GA channels)
        # vol_label_names_map was built during the per-subject label mapping
        # and maps integer parcel IDs to human-readable names.
        # NOTE: MNE appends '-0', '-1', etc. suffixes to channel names when
        # duplicates exist (e.g. multiple centroids mapping to 'parcel_0').
        # We strip these suffixes before matching to atlas label names.
        import re
        ga_ch_names = grand_avg[feature].info['ch_names']
        # Build index from base name (suffix stripped) → first matching channel index
        ga_base_to_idx = {}
        for idx, ch_name in enumerate(ga_ch_names):
            base_name = re.sub(r'-\d+$', '', ch_name)
            if base_name not in ga_base_to_idx:
                ga_base_to_idx[base_name] = idx

        # Map: parcel ID → GA channel index (via label name)
        parcel_to_ga_idx = {}
        if vol_label_names_map is not None:
            for p_id, p_name in vol_label_names_map.items():
                if p_name in ga_base_to_idx:
                    parcel_to_ga_idx[p_id] = ga_base_to_idx[p_name]
        else:
            # Custom atlas with no label names: channel names are "parcel_<id>"
            for p_id in np.unique(fsavg_parcel_ids):
                p_name = f'parcel_{p_id}'
                if p_name in ga_base_to_idx:
                    parcel_to_ga_idx[p_id] = ga_base_to_idx[p_name]

        # --- Diagnostics ---
        print(f'  GA diagnostics: {len(parcel_to_ga_idx)} parcel-to-channel matches '
              f'(from {len(np.unique(fsavg_parcel_ids[fsavg_parcel_ids != 0]))} parcels, '
              f'{len(ga_ch_names)} GA channels)')
        # --- End diagnostics ---

        # Fill all vertices with their parcel's GA TRF value
        full_vertno_ga = src_fsavg[0]['vertno']
        full_data_ga = np.zeros((len(full_vertno_ga), grand_avg[feature].data.shape[1]))
        for v_idx, p_id in enumerate(fsavg_parcel_ids):
            if p_id in parcel_to_ga_idx:
                full_data_ga[v_idx] = grand_avg[feature].data[parcel_to_ga_idx[p_id]]

        stc_ga_vol = mne.VolSourceEstimate(
            data=full_data_ga,
            vertices=[full_vertno_ga],
            tmin=grand_avg[feature].times[0],
            tstep=1 / grand_avg[feature].info['sfreq']
        )
        for it in feat_initial_times:
            it_suffix = f'_{it}s' if it is not None else ''
            brain = plot_general.sources(
                stc=stc_ga_vol, src=src_fsavg, subject='fsaverage',
                subjects_dir=subjects_dir, initial_time=it,
                surf_vol='volume', force_fsaverage=True,
                source_estimation='trf', views=['lateral', 'medial'],
                alpha=0.5, plot_margin=plot_margin,
                save_fig=save_fig, fig_path=fig_path,
                fname=f'{feature}_GA_source_trf_brain{it_suffix}')

# Save grand average
if save_data:
    save.var(var=grand_avg, path=save_path_trf, fname='GA_grand_avg.pkl')
    save.var(var=feature_evokeds, path=save_path_trf, fname='GA_all_subjects.pkl')

# --------- Statistics: cluster-based permutation test per feature ---------#
# One-sample (vs. zero) spatio-temporal cluster test on the TRF coefficients,
# run independently for each feature. Spatial clustering uses the region
# adjacency (anatomical borders for surface parcellations, geometric centroid
# proximity for volume parcellations) so contiguous regions form clusters.
if run_permutations:
    import pandas as pd

    print(f"\n{'='*60}")
    print("Running cluster-based permutation statistics (per feature)...")
    print(f"{'='*60}")

    # Regions present in EVERY subject and feature, in grand-average channel order
    channel_sets = [set(ev.ch_names) for ev_list in feature_evokeds.values() for ev in ev_list]
    common = set.intersection(*channel_sets)
    stat_regions = [ch for ch in grand_avg[features[0]].ch_names if ch in common]

    # Region adjacency (computed once; identical region set for all features)
    region_adjacency = functions_analysis.get_parcellation_adjacency(
        ch_names=stat_regions, surf_vol=surf_vol, subject='fsaverage',
        subjects_dir=subjects_dir, parc=parc,
        label_positions=ga_label_positions if ga_label_positions else label_positions)

    # For volume parcellations, pre-compute the fsaverage volume → GA-channel
    # mapping once (feature-independent) so significance-masked volume brains can
    # be rendered. Reuses the atlas variables built in the per-subject loop
    # (atlas_data, inv_atlas_affine, mni305_to_mni152, vol_label_names_map).
    vol_sig_ready = False
    if plot_significance and surf_vol == 'vol_parcellation':
        try:
            import re
            fname_src_fsavg_sig = (paths.sources_path + 'fsaverage'
                                   + f'/fsaverage_volume_{spacing}_{int(pos)}-src.fif')
            src_fsavg_sig = mne.read_source_spaces(fname_src_fsavg_sig)

            fsavg_inuse_sig = src_fsavg_sig[0]['inuse'].astype(bool)
            fsavg_rr_mm_sig = src_fsavg_sig[0]['rr'][fsavg_inuse_sig] * 1000
            ras_mni_t_sig = read_ras_mni_t('fsaverage', subjects_dir)
            fsavg_mni305_sig = apply_trans(ras_mni_t_sig, fsavg_rr_mm_sig)
            fsavg_mni152_sig = apply_trans(mni305_to_mni152, fsavg_mni305_sig)
            fsavg_vox_sig = apply_trans(inv_atlas_affine, fsavg_mni152_sig)
            fsavg_vox_idx_sig = np.round(fsavg_vox_sig).astype(int)
            for dim in range(3):
                fsavg_vox_idx_sig[:, dim] = np.clip(fsavg_vox_idx_sig[:, dim], 0, atlas_data.shape[dim] - 1)
            fsavg_parcel_ids_sig = atlas_data[fsavg_vox_idx_sig[:, 0], fsavg_vox_idx_sig[:, 1], fsavg_vox_idx_sig[:, 2]]

            # parcel ID → GA channel index (via label name, suffixes stripped)
            ga_ref_names = grand_avg[features[0]].info['ch_names']
            ga_base_to_idx_sig = {}
            for idx, ch_name in enumerate(ga_ref_names):
                base_name = re.sub(r'-\d+$', '', ch_name)
                if base_name not in ga_base_to_idx_sig:
                    ga_base_to_idx_sig[base_name] = idx

            parcel_to_ga_idx_sig = {}
            if vol_label_names_map is not None:
                for p_id, p_name in vol_label_names_map.items():
                    if p_name in ga_base_to_idx_sig:
                        parcel_to_ga_idx_sig[p_id] = ga_base_to_idx_sig[p_name]
            else:
                for p_id in np.unique(fsavg_parcel_ids_sig):
                    p_name = f'parcel_{p_id}'
                    if p_name in ga_base_to_idx_sig:
                        parcel_to_ga_idx_sig[p_id] = ga_base_to_idx_sig[p_name]

            full_vertno_ga_sig = src_fsavg_sig[0]['vertno']
            vol_sig_ready = True
        except Exception as e:
            print(f'  Could not prepare volume significance mapping, skipping volume sig plots: {e}')

    clusters_mask = {}
    clusters_pvalues = {}
    sig_rows = []

    for feature in features:
        print(f'  Permutation test for feature: {feature}')
        # data shape: (subjects, times, regions)
        data = np.array([ev.copy().pick(stat_regions).data.T for ev in feature_evokeds[feature]])
        mask_tr, clusters_pvalues[feature] = functions_analysis.run_permutations_test(
            data=data, pval_threshold=pval_threshold, t_thresh=t_thresh,
            adj_matrix=region_adjacency, n_permutations=n_permutations)
        mask_rt = mask_tr.T  # (regions, times)
        clusters_mask[feature] = mask_rt

        # Summarize significant regions and their significant time windows
        times = grand_avg[feature].times
        for r_idx, region in enumerate(stat_regions):
            sig_times = times[mask_rt[r_idx]]
            if sig_times.size:
                sig_rows.append({'feature': feature, 'region': region,
                                 'n_sig_times': int(sig_times.size),
                                 't_start': float(sig_times.min()),
                                 't_end': float(sig_times.max())})

        # Significance-masked GA brain plot (surface and volume parcellations)
        if plot_significance and mask_rt.any():

            feat_initial_times = functions_analysis.get_feature_initial_time(feature, initial_time)

            # Full-channel mask aligned to grand_avg[feature] channel order
            ga_full = grand_avg[feature]
            full_idx = {ch: i for i, ch in enumerate(ga_full.ch_names)}
            full_mask = np.zeros(ga_full.data.shape, dtype=bool)
            for r_idx, region in enumerate(stat_regions):
                if region in full_idx:
                    full_mask[full_idx[region]] = mask_rt[r_idx]
            masked_full_data = ga_full.data * full_mask  # zero non-significant region/time

            if surf_vol == 'parcellation':
                ga_sig = ga_full.copy()
                ga_sig.data = masked_full_data
                stc_sig, src_sig = functions_analysis.evoked_to_parcellation_stc(
                    ga_sig, parc, 'fsaverage', subjects_dir, spacing)
                for it in feat_initial_times:
                    it_suffix = f'_{it}s' if it is not None else ''
                    plot_general.sources(
                        stc=stc_sig, src=src_sig, subject='fsaverage',
                        subjects_dir=subjects_dir, initial_time=it,
                        surf_vol='surface', force_fsaverage=True,
                        source_estimation='trf', views=['lateral', 'medial'],
                        plot_margin=plot_margin,
                        save_fig=save_fig, fig_path=fig_path,
                        fname=f'{feature}_GA_source_trf_sig{it_suffix}')

            elif surf_vol == 'vol_parcellation' and vol_sig_ready:
                # Fill all fsaverage volume vertices with their parcel's masked GA value
                full_data_ga_sig = np.zeros((len(full_vertno_ga_sig), masked_full_data.shape[1]))
                for v_idx, p_id in enumerate(fsavg_parcel_ids_sig):
                    if p_id in parcel_to_ga_idx_sig:
                        full_data_ga_sig[v_idx] = masked_full_data[parcel_to_ga_idx_sig[p_id]]

                stc_sig_vol = mne.VolSourceEstimate(
                    data=full_data_ga_sig,
                    vertices=[full_vertno_ga_sig],
                    tmin=grand_avg[feature].times[0],
                    tstep=1 / grand_avg[feature].info['sfreq'])

                for it in feat_initial_times:
                    it_suffix = f'_{it}s' if it is not None else ''
                    plot_general.sources(
                        stc=stc_sig_vol, src=src_fsavg_sig, subject='fsaverage',
                        subjects_dir=subjects_dir, initial_time=it,
                        surf_vol='volume', force_fsaverage=True,
                        source_estimation='trf', views=['lateral', 'medial'],
                        alpha=0.5, plot_margin=plot_margin,
                        save_fig=save_fig, fig_path=fig_path,
                        fname=f'{feature}_GA_source_trf_sig{it_suffix}')

        elif plot_significance:
            # No significant region/time survived the cluster test -> no _sig figure
            print(f'  No significant clusters for feature "{feature}" '
                  f'(p < {pval_threshold}); skipping significance figure.')

    # Save cluster masks / p-values
    if save_data:
        save.var(var={'clusters_mask': clusters_mask,
                      'clusters_pvalues': clusters_pvalues,
                      'stat_regions': stat_regions,
                      'pval_threshold': pval_threshold,
                      't_thresh': t_thresh,
                      'n_permutations': n_permutations},
                 path=save_path_trf, fname='GA_stats_clusters.pkl')

    # Save significant-regions summary CSV
    os.makedirs(fig_path, exist_ok=True)
    sig_df = pd.DataFrame(sig_rows, columns=['feature', 'region', 'n_sig_times', 't_start', 't_end'])
    sig_df.to_csv(fig_path + 'significant_regions.csv', index=False)
    print(f'  Saved significance summary ({len(sig_rows)} region hits): '
          f'{fig_path}significant_regions.csv')

print(f"\nSource parcellation TRF analysis completed!")
print(f"Results saved to: {save_path_trf.split(paths.save_path)[-1]}")
