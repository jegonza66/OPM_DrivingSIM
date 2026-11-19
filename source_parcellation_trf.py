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
from mne.transforms import apply_trans


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
              'band_id': (0.1, 40),
              'data_type': 'processed',
              'filter_sensors': True,
              }

# Source estimation parameters
surf_vol = 'parcellation'  # 'parcellation' | 'vol_parcellation'
parc = 'aparc.a2009s'          # Surface parcellation (used when surf_vol='parcellation')
# Volume atlas (used when surf_vol='vol_parcellation') — must match sourcemodel_setup.py
# 'aal3' (AAL3v2) | 'aal' (AAL SPM12) | 'destrieux' | 'harvard_oxford' | 'schaefer' | or path to a .nii(.gz) atlas.
# For a custom NIfTI you may place a sibling "<name>_labels.txt" (lines "<id> <name>")
# to get human-readable region names (e.g. MarsAtlas below).
# vol_parc_atlas = os.path.join(paths.atlas_path, 'colin27_MNI_MarsAtlas.nii')  # MarsAtlas (Colin27/MNI)
vol_parc_atlas = 'aal3'
pick_ori = None
spacing = 'ico4'  # Must match sourcemodel_setup.py setting
pos = 10          # Must match sourcemodel_setup.py setting (volume grid spacing in mm)

# TRF parameters
trf_params = {
    'input_features': {
        # Phase pieces only: the unsplit 'fix'/'sac' would be (near-)collinear with their sum.
        # Common response and phase effects are computed post hoc from these.
        'fix_CF': None, 'fix_DA': None, 'fix_Audio': None,
        'sac_CF': None, 'sac_DA': None, 'sac_Audio': None,
        # 'audio_env_std_Audio' dropped: DA1 has no audio, and the shared-regularization fit needs the same features for all
        'Steering_std_der': None,
        'Gas_std_der': None,
        'Brake_std_der': None,
        'left_but': None,
        'right_but': None,
    },
    # Same method as MEGEYEDYN: one joint TRFlearn model per subject, ONE ridge regularization shared by
    # all subjects (median of each subject's R²-best). Unstandardized impulses; kernels on standardized parcels.
    'regularization': 'median',
    'regularization_grid': np.logspace(-9, 3, 13),
    # Per-feature duration: use dict with 'default' key and optional per-feature overrides
    # e.g. 'tmin': {'default': -0.2, 'left_but': -2}, 'tmax': {'default': 0.5, 'left_but': 2}
    'tmin': {'default': -0.2, 'Steering_std_der': -2, 'left_but': -2, 'right_but': -2, 'Gas_std_der': -2, 'Brake_std_der': -2},
    'tmax': {'default': 0.5, 'Steering_std_der': 2, 'left_but': 2, 'right_but': 2, 'Gas_std_der': 2, 'Brake_std_der': 2},
    'plot_margin': 0.15,  # seconds to crop from each side of plotted TRF time series
}
# Same fit in both; 'original' rescales each subject's kernels by its parcel SDs (LCMV a.u. per event)
kernel_units_list = ['standardized', 'original']
# A feature's group average and test only take subjects with at least this many of its events
# (DA1 has no Audio phase, so its fix_Audio / sac_Audio regressors are empty)
min_subject_events = 1

# Per-feature initial time for brain plots: dict with 'default' key and optional per-feature overrides
# e.g. {'default': None, 'fix': 0.1, 'left_but': 0.3}
initial_time = {'default': None,
                'fix': None,
                'sac': [0.0, 0.12],
                'sac_CF': [0.0, 0.12], 'sac_DA': [0.0, 0.12], 'sac_Audio': [0.0, 0.12],
                'pur': None,
                'audio_env_std_Audio': 0.0,
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
trf_params['features'] = features

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
fig_path = paths.plots_path + (f"TRF_Source_{meg_params['data_type']}/Band_{meg_params['band_id']}/{parc_tag}/"
                               f"{functions_general.features_path_str(trf_params['input_features'])}_{_path_tmin}_{_path_tmax}_"
                               f"trflearn_{trf_params['regularization']}_ori{pick_ori}/{meg_params['chs_id']}/").replace(":", "")
save_path_trf = fig_path.replace(paths.plots_path, paths.save_path)

# --------- Run ---------#
# Loop 1: parcel time courses + regressors per subject (kept for the group fit)
subjects = []
subject_inputs = {}
subject_meta = {}
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

    # The saved forward is free-orientation. pick_ori='normal' requires a
    # surface-oriented forward (dipoles expressed in surface-normal coords),
    # so convert it here. 'normal' is only defined for surface source spaces;
    # volume parcellations have no surface normal.
    if pick_ori == 'normal':
        if surf_vol != 'parcellation':
            raise ValueError(
                "pick_ori='normal' is only valid for surface source spaces "
                f"(surf_vol='parcellation'), not surf_vol='{surf_vol}'. "
                "Use 'max-power' (scalar, sign-preserving) or None instead.")
        fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=False,
                                           use_cps=True)

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

        # Use the vertices actually returned by the beamformer (stc.vertices),
        # not src['vertno']. If two labels resolve to the same centroid vertex,
        # the source space can hold a duplicate that the beamformer collapses,
        # leaving source_data with one fewer row than src['vertno']. stc.vertices
        # is guaranteed to align row-for-row with source_data.
        lh_verts = stc.vertices[0]
        rh_verts = stc.vertices[1]

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
            # Try to load a companion labels file mapping integer ID -> name.
            # Looks for "<atlas_basename>_labels.txt" next to the .nii(.gz),
            # then a generic "<atlas_basename>.txt". Each non-comment line is
            # "<id> <name>" (whitespace- or comma-separated). If none is found,
            # fall back to integer IDs.
            vol_label_names_map = None
            _atlas_base = vol_parc_atlas
            for _ext in ('.nii.gz', '.nii'):
                if _atlas_base.endswith(_ext):
                    _atlas_base = _atlas_base[:-len(_ext)]
                    break
            for _labels_fname in (_atlas_base + '_labels.txt', _atlas_base + '.txt'):
                if os.path.isfile(_labels_fname):
                    vol_label_names_map = {}
                    with open(_labels_fname, 'r') as _lf:
                        for _line in _lf:
                            _line = _line.strip()
                            if not _line or _line.startswith('#'):
                                continue
                            _parts = _line.replace(',', ' ').split()
                            try:
                                _lid = int(_parts[0])
                            except (ValueError, IndexError):
                                continue
                            _lname = ' '.join(_parts[1:]) if len(_parts) > 1 else f'parcel_{_lid}'
                            vol_label_names_map[_lid] = _lname
                    print(f'Loaded {len(vol_label_names_map)} region names from '
                          f'{os.path.basename(_labels_fname)}')
                    break
        else:
            from nilearn import datasets as ni_datasets
            # Pin AAL versions: nilearn's default changed from SPM12 (116 regions) to AAL3 (3v2)
            if vol_parc_atlas == 'aal':
                atlas = ni_datasets.fetch_atlas_aal(version='SPM12')
            elif vol_parc_atlas == 'aal3':
                atlas = ni_datasets.fetch_atlas_aal(version='3v2')
            elif vol_parc_atlas == 'destrieux':
                atlas = ni_datasets.fetch_atlas_destrieux_2009()
            elif vol_parc_atlas == 'harvard_oxford':
                atlas = ni_datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
            elif vol_parc_atlas == 'schaefer':
                atlas = ni_datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
            else:
                raise ValueError(f'Unknown vol_parc_atlas: {vol_parc_atlas}')
            # `atlas['maps']` may be a file path or an already-loaded image
            # depending on the nilearn version / atlas.
            atlas_maps = atlas['maps']
            atlas_img = atlas_maps if isinstance(atlas_maps, nib.spatialimages.SpatialImage) \
                else nib.load(atlas_maps)
            # Build mapping: integer ID → label name. nilearn returns `labels`
            # in different shapes depending on the atlas / version:
            #   - plain list of names (AAL, Schaefer, Harvard-Oxford)
            #   - numpy structured/record array with 'index'+'name' fields
            #   - pandas DataFrame with 'index'+'name' columns (Destrieux 2009)
            # Normalize to (ids, names) and prefer explicit ids when available.
            atlas_data_full = np.asarray(atlas_img.dataobj).astype(int)
            atlas_label_list = atlas.get('labels', None)
            atlas_indices = atlas.get('indices', None)  # AAL provides this
            vol_label_names_map = None

            if atlas_label_list is not None:
                ids = None
                names = None

                def _to_str(x):
                    return x.decode() if isinstance(x, bytes) else str(x)

                # pandas DataFrame (e.g. Destrieux 2009)
                try:
                    import pandas as pd
                    if isinstance(atlas_label_list, pd.DataFrame):
                        cols = {str(c).lower(): c for c in atlas_label_list.columns}
                        name_col = cols.get('name', list(atlas_label_list.columns)[-1])
                        names = [_to_str(v) for v in atlas_label_list[name_col].tolist()]
                        if 'index' in cols:
                            ids = [int(v) for v in atlas_label_list[cols['index']].tolist()]
                except Exception:
                    pass

                # numpy structured / record array with named fields
                if names is None and getattr(getattr(atlas_label_list, 'dtype', None), 'names', None):
                    fields = atlas_label_list.dtype.names
                    name_field = 'name' if 'name' in fields else fields[-1]
                    names = [_to_str(v) for v in atlas_label_list[name_field]]
                    if 'index' in fields:
                        ids = [int(v) for v in atlas_label_list['index']]

                # plain list / 1-D array of names
                if names is None:
                    names = [_to_str(v) for v in list(atlas_label_list)]

                # explicit volume IDs (e.g. AAL `indices`)
                if ids is None and atlas_indices is not None:
                    try:
                        ids = [int(v) for v in atlas_indices]
                    except Exception:
                        ids = None

                unique_ids = sorted(int(u) for u in np.unique(atlas_data_full[atlas_data_full != 0]))

                if ids is not None and len(ids) == len(names):
                    # Direct id → name mapping (most robust)
                    vol_label_names_map = {int(i): n for i, n in zip(ids, names)}
                elif len(names) == len(unique_ids):
                    vol_label_names_map = {uid: names[i] for i, uid in enumerate(unique_ids)}
                elif len(names) == len(unique_ids) + 1:
                    # names include a background entry as the first element
                    vol_label_names_map = {uid: names[i + 1] for i, uid in enumerate(unique_ids)}
                else:
                    vol_label_names_map = {uid: (names[i] if i < len(names) else f'parcel_{uid}')
                                           for i, uid in enumerate(unique_ids)}

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
        # Use stc.vertices (beamformer output) so label_names aligns with source_data.
        verts = stc.vertices[0]
        vert_rr_mm = src[0]['rr'][verts] * 1000  # m → mm (surface-RAS)

        ras_mni_t = functions_general.tkr_to_mni305_mm(subject_code, subjects_dir)
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
    input_arrays = functions_analysis.build_trf_inputs(subject=subject, meg_data=meg_data, features=features,
                                                       meg_params=meg_params, use_saved_data=use_saved_data)
    subjects.append(subject_id)
    subject_inputs[subject_id] = (raw_src, input_arrays)

    # --------- Compute label positions for spatial coloring ---------#
    # Key positions by the ACTUAL (de-duplicated) channel names. When several
    # source centroids share a parcel name, mne.create_info appends '-0', '-1',
    # ... to make them unique, so the Evoked/grand-average channels carry those
    # suffixes. Using raw_src channel names keeps the keys aligned with the data
    # and preserves each centroid's own position (instead of collapsing
    # duplicates onto a single base-name key).
    label_positions = {}
    src_ch_names = raw_src.info['ch_names']
    if surf_vol == 'parcellation':
        # Index rr by the beamformer-returned vertices (stc.vertices) so the
        # positions stay aligned with source_data / raw_src channels (see note
        # in the label-name mapping above about collapsed duplicate centroids).
        lh_verts_pos = src[0]['rr'][stc.vertices[0]]
        rh_verts_pos = src[1]['rr'][stc.vertices[1]]
        all_pos = np.vstack([lh_verts_pos, rh_verts_pos])
        for idx_lp, name_lp in enumerate(src_ch_names):
            label_positions[name_lp] = all_pos[idx_lp]
    elif surf_vol == 'vol_parcellation':
        vol_verts_pos = src[0]['rr'][stc.vertices[0]]
        for idx_lp, name_lp in enumerate(src_ch_names):
            label_positions[name_lp] = vol_verts_pos[idx_lp]

    subject_meta[subject_id] = (subject_code, src, label_positions)


# --------- Group TRF fit (shared regularization) ---------#
# Kernels are cached per subject and reused only when every subject is there
kernels, n_events, target_scales = functions_analysis.cached_trf_group(
    subjects, lambda subject_id: subject_inputs.pop(subject_id), trf_params, save_path_trf + 'kernels/',
    overwrite=not use_saved_data)
subject_inputs.clear()
feature_subjects = {feature: [s for s in subjects if n_events[s][feature] >= min_subject_events] for feature in features}
plot_margin = trf_params.get('plot_margin', 0)
base_fig_path, base_save_path_trf = fig_path, save_path_trf

for units in kernel_units_list:
    fig_path = base_fig_path + f'{units}/'
    save_path_trf = base_save_path_trf + f'{units}/'
    feature_evokeds = {feature: [] for feature in features}

    # Loop 2: per-subject TRF evokeds and individual plots
    for subject_id in subjects:
        subject_code, src, label_positions = subject_meta[subject_id]
        for feature in features:
            if subject_id not in feature_subjects[feature]:
                continue
            feat_tmin, feat_tmax = functions_analysis.get_feature_tmin_tmax(feature, trf_params)
            subj_evoked = functions_analysis.kernel_units(kernels[subject_id][feature], target_scales[subject_id], units)
            # whole-window baseline, as before (and as MEGEYEDYN)
            subj_evoked.apply_baseline((None, None))
            trf_coefs = subj_evoked.data  # (n_sources, n_delays)
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

                    ras_mni_t_plot = functions_general.tkr_to_mni305_mm(subject_code, subjects_dir)
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
                        tstep=1 / subj_evoked.info['sfreq']
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
            ras_mni_t_ga = functions_general.tkr_to_mni305_mm('fsaverage', subjects_dir)
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
        # Subject IDs per feature, in the same order as GA_all_subjects (features can have different subjects)
        save.var(var=feature_subjects, path=save_path_trf, fname='GA_subjects.pkl')

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
                ras_mni_t_sig = functions_general.tkr_to_mni305_mm('fsaverage', subjects_dir)
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
