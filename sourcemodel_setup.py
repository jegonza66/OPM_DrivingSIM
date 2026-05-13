import os
import mne
import numpy as np
import nibabel as nib
import functions_general
import paths
import load
import setup
import plot_general
from mne.transforms import apply_trans, read_ras_mni_t

# Load experiment info
exp_info = setup.exp_info()


# --------- Setup ---------#
task = 'DA2'
# Define surface, volume, mixed, or parcellation source space
chs_id = 'mag_z'
surf_vol = 'vol_parcellation'  # 'surface' | 'volume' | 'mixed' | 'parcellation' | 'vol_parcellation'
force_fsaverage = False
spacing = 'ico4'
pos = 10
pick_ori = None # 'normal' | 'max-power' | None
depth = None
parc = 'aparc.a2009s'  # Parcellation atlas (used when surf_vol='parcellation')

# Volumetric atlas in MNI152 space (used when surf_vol='vol_parcellation')
# Option A: Use a local .nii.gz file
# vol_parc_path = os.path.join(paths.mri_path, 'atlases', 'atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz')
# Option B: Use a standard atlas fetched via nilearn (e.g. 'aal', 'destrieux', 'harvard_oxford')
vol_parc_atlas = 'aal'  # 'aal' | 'destrieux' | 'harvard_oxford' | 'schaefer' | or path to a .nii.gz file

meg_params = {'data_type': 'processed'}

# Define Subjects_dir as Freesurfer output folder
mri_path = paths.mri_path
subjects_dir = os.path.join(mri_path, 'freesurfer')
os.environ["SUBJECTS_DIR"] = subjects_dir

# Digitalization data path
dig_path = paths.opt_path
visualize_alignment = False


# --------- Coregistration ---------#

# Iterate over subjects
for subject_id in exp_info.subjects_ids + ['fsaverage']:
# for subject_id in ['18630']:

    if subject_id != 'fsaverage':

        # Load subject
        subject = setup.subject(subject_id=subject_id)

        # Load MEG
        meg_data = load.meg(subject_id=subject_id, meg_params=meg_params)
        picks = functions_general.pick_chs(chs_id=chs_id, info=meg_data.info)
        meg_data.pick(picks)

        if force_fsaverage:
            subject_code = 'fsaverage'
            dig = False
            # Check mean distances if already run transformation
            # This should actually load the fsaverage transformation
            trans_path = os.path.join(paths.mri_path, 'freesurfer', subject_id, 'bem', f'{subject_id}-trans.fif')
            trans = mne.read_trans(trans_path)
            print('Distance from head origin to MEG origin: %0.1f mm'
                  % (1000 * np.linalg.norm(meg_data.info['dev_head_t']['trans'][:3, 3])))
            print('Distance from head origin to MRI origin: %0.1f mm'
                  % (1000 * np.linalg.norm(trans['trans'][:3, 3])))

        else:
            subject_code = subject_id
            dig = True
            # Check if subject has MRI data
            fs_subj_path = os.path.join(subjects_dir, subject_id)

            if len(os.listdir(fs_subj_path)):
                try:
                    # Check mean distances if already run transformation
                    trans_path = os.path.join(paths.mri_path, 'freesurfer', subject_id, 'bem', f'{subject_id}-trans.fif')
                    trans = mne.read_trans(trans_path)
                    print('Distance from head origin to MEG origin: %0.1f mm'
                          % (1000 * np.linalg.norm(meg_data.info['dev_head_t']['trans'][:3, 3])))
                    print('Distance from head origin to MRI origin: %0.1f mm'
                          % (1000 * np.linalg.norm(trans['trans'][:3, 3])))
                except:
                    # Make info object
                    dig_info = meg_data.pick('meg').info.copy()
                    # dig_info.set_montage(montage=dig_montage)

                    # Save raw instance with info
                    info_raw = mne.io.RawArray(np.zeros((dig_info['nchan'], 1)), dig_info)
                    dig_path_subject = dig_path + subject_id
                    dig_info_path = dig_path_subject + '/info_raw.fif'
                    info_raw.save(dig_info_path, overwrite=True)

                    # Align and save fiducials and transformation files to FreeSurfer/subject/bem folder
                    mne.gui.coregistration(subject=subject_id, subjects_dir=subjects_dir, inst=dig_info_path,
                                           block=True)

            # If subject has no MRI data
            else:
                subject_code = 'fsaverage'
                dig = False
                # Check mean distances if already run transformation
                trans_path = os.path.join(paths.mri_path, 'freesurfer', subject_id, 'bem', f'{subject_id}-trans.fif')
                trans = mne.read_trans(trans_path)
                print('Distance from head origin to MEG origin: %0.1f mm'
                      % (1000 * np.linalg.norm(meg_data.info['dev_head_t']['trans'][:3, 3])))
                print('Distance from head origin to MRI origin: %0.1f mm'
                      % (1000 * np.linalg.norm(trans['trans'][:3, 3])))
    else:
        subject_code = 'fsaverage'
        dig = False

    if visualize_alignment:
        # Load subject
        subject = setup.subject(subject_id=subject_id)
        plot_general.mri_meg_alignment(subject=subject, subject_code=subject_code, dig=dig, subjects_dir=subjects_dir)

    # --------- Bem model ---------#
    # Source data and models path
    sources_path_subject = paths.sources_path + subject_id
    os.makedirs(sources_path_subject, exist_ok=True)
    os.makedirs(paths.sources_path + subject_code, exist_ok=True)

    fname_bem = paths.sources_path + subject_code + f'/{subject_code}_bem_{spacing}-sol.fif'
    try:
        # Load
        bem = mne.read_bem_solution(fname_bem)
    except:
        # Compute
        model = mne.make_bem_model(subject=subject_code, ico=int(spacing[-1]), conductivity=[0.3], subjects_dir=subjects_dir)
        bem = mne.make_bem_solution(model)
        # Save
        mne.write_bem_solution(fname_bem, bem, overwrite=True)

    # --------- Source space, forward model and inverse operator ---------#
    if surf_vol == 'volume':
        # Volume
        # Source model
        fname_src = paths.sources_path + subject_code + f'/{subject_code}_volume_{spacing}_{int(pos)}-src.fif'
        try:
            # Load
            src = mne.read_source_spaces(fname_src)
        except:
            # Compute
            src = mne.setup_volume_source_space(subject=subject_code, subjects_dir=subjects_dir, bem=bem, pos=pos,
                                                sphere_units='m', add_interpolator=True)
            # Save
            mne.write_source_spaces(fname_src, src, overwrite=True)

        # Forward model
        if subject_id != 'fsaverage':
            fname_fwd = sources_path_subject + f'/{subject_code}_{meg_params['data_type']}_chs{chs_id}_volume_{spacing}_{int(pos)}-fwd.fif'
            # Compute
            fwd = mne.make_forward_solution(meg_data.info, trans=trans_path, src=src, bem=bem)
            mne.write_forward_solution(fname_fwd, fwd, overwrite=True)

    elif surf_vol == 'surface':
        # Source model
        fname_src = paths.sources_path + subject_code + f'/{subject_code}_surface_{spacing}-src.fif'
        try:
            # Load
            src = mne.read_source_spaces(fname_src)
        except:
            # Compute
            src = mne.setup_source_space(subject=subject_code, spacing=spacing, subjects_dir=subjects_dir)
            # Save
            mne.write_source_spaces(fname_src, src, overwrite=True)

        # Forward model
        if subject_id != 'fsaverage':
            fname_fwd = sources_path_subject + f'/{subject_code}_{meg_params['data_type']}_chs{chs_id}_surface_{spacing}-fwd.fif'
            fwd = mne.make_forward_solution(meg_data.info, trans=trans_path, src=src, bem=bem)
            mne.write_forward_solution(fname_fwd, fwd, overwrite=True)

    elif surf_vol == 'mixed':
        fname_src_mix = paths.sources_path + subject_code + f'/{subject_code}_mixed_{spacing}_{int(pos)}-src.fif'
        try:
            # Load
            src_surf = mne.read_source_spaces(fname_src_surf)
        except:
            # Mixed
            # Surface source model
            fname_src_surf = paths.sources_path + subject_code + f'/{subject_code}_surface_{spacing}-src.fif'
            try:
                # Load
                src_surf = mne.read_source_spaces(fname_src_surf)
            except:
                # Compute
                src_surf = mne.setup_source_space(subject=subject_code, spacing=spacing, subjects_dir=subjects_dir)
                # Save
                mne.write_source_spaces(fname_src_surf, src_surf, overwrite=True)

            # Volume source model
            fname_src_vol = paths.sources_path + subject_code + f'/{subject_code}_volume_{spacing}_{int(pos)}-src.fif'
            # Compute
            src_vol = mne.setup_volume_source_space(subject=subject_code, subjects_dir=subjects_dir, bem=bem, pos=pos, sphere_units='m', add_interpolator=True)
            # Save
            mne.write_source_spaces(fname_src_vol, src_vol, overwrite=True)

            # Mixed source space
            src = src_surf + src_vol
            # Save
            mne.write_source_spaces(fname_src_mix, src, overwrite=True)

        # Forward model
        if subject_id != 'fsaverage':
            fwd = mne.make_forward_solution(meg_data.info, trans=trans_path, src=src, bem=bem)
            fname_fwd = sources_path_subject + f'/{subject_code}_{meg_params['data_type']}_chs{chs_id}_mixed_{spacing}_{int(pos)}-fwd.fif'
            mne.write_forward_solution(fname_fwd, fwd, overwrite=True)

    elif surf_vol == 'parcellation':
        # Parcellation-based source model with dipoles only at label centroids.
        # Yields a low-dimensional source space (~150 sources for aparc.a2009s)
        # suitable for TRF analysis on continuous data.

        # First, ensure a surface source space exists (needed to define valid vertices)
        fname_src_surf = paths.sources_path + subject_code + f'/{subject_code}_surface_{spacing}-src.fif'
        try:
            src_surf = mne.read_source_spaces(fname_src_surf)
        except:
            src_surf = mne.setup_source_space(subject=subject_code, spacing=spacing, subjects_dir=subjects_dir)
            mne.write_source_spaces(fname_src_surf, src_surf, overwrite=True)

        # Read parcellation labels (needed for both source space and forward)
        labels = mne.read_labels_from_annot(subject_code, parc=parc, subjects_dir=subjects_dir)
        labels = [l for l in labels if 'unknown' not in l.name.lower() and 'medialwall' not in l.name.lower()]

        # Parcellation source space
        fname_src = paths.sources_path + subject_code + f'/{subject_code}_parcellation_{parc}-src.fif'
        try:
            src = mne.read_source_spaces(fname_src)
        except:
            # Make a copy of the surface source space to restrict
            src = src_surf.copy()

            # Find centroid vertex for each label, restricted to source space vertices
            centroid_vertices = {0: [], 1: []}
            for label in labels:
                hemi_idx = 0 if label.hemi == 'lh' else 1
                centroid = label.center_of_mass(subject=subject_code, subjects_dir=subjects_dir,
                                                restrict_vertices=src)
                centroid_vertices[hemi_idx].append(centroid)

            # Restrict source space to only centroid vertices
            for hemi_idx in range(2):
                verts = np.array(sorted(centroid_vertices[hemi_idx]))
                src[hemi_idx]['inuse'][:] = 0
                src[hemi_idx]['inuse'][verts] = 1
                src[hemi_idx]['nuse'] = len(verts)
                src[hemi_idx]['vertno'] = verts
                src[hemi_idx]['use_tris'] = np.array([], dtype=int).reshape(0, 3)
                src[hemi_idx]['nuse_tri'] = 0

            n_sources = sum(len(v) for v in centroid_vertices.values())
            print(f'Parcellation source space: {n_sources} sources ({parc})')

            # Save
            mne.write_source_spaces(fname_src, src, overwrite=True)

        # Forward model
        if subject_id != 'fsaverage':
            fname_fwd = sources_path_subject + f'/{subject_code}_{meg_params['data_type']}_chs{chs_id}_parcellation_{parc}-fwd.fif'

            # Compute forward from the full surface source space, then restrict
            # to centroid vertices. Direct forward on sparse centroid source spaces
            # can fail when vertices near the inner skull BEM surface are excluded.
            fname_fwd_surf = sources_path_subject + f'/{subject_code}_{meg_params['data_type']}_chs{chs_id}_surface_{spacing}-fwd.fif'
            try:
                fwd_surf = mne.read_forward_solution(fname_fwd_surf)
            except:
                fwd_surf = mne.make_forward_solution(meg_data.info, trans=trans_path, src=src_surf, bem=bem)
                mne.write_forward_solution(fname_fwd_surf, fwd_surf, overwrite=True)

            # Create single-vertex labels at each parcellation centroid
            centroid_labels = []
            for label in labels:
                centroid = label.center_of_mass(subject=subject_code, subjects_dir=subjects_dir,
                                                restrict_vertices=src_surf)
                centroid_labels.append(mne.Label([centroid], hemi=label.hemi,
                                                name=label.name, subject=subject_code))

            # Restrict forward to centroid vertices only
            fwd = mne.forward.restrict_forward_to_label(fwd_surf, centroid_labels)
            mne.write_forward_solution(fname_fwd, fwd, overwrite=True)

    elif surf_vol == 'vol_parcellation':
        # ---------------------------------------------------------------
        # Volume parcellation using a NIfTI atlas defined in MNI152 space
        # ---------------------------------------------------------------
        # Each subject's volume source positions (in FreeSurfer surface-RAS)
        # are transformed to MNI152 so we can look up atlas parcel IDs:
        #
        #   surface-RAS  →  MNI305          per-subject, via FreeSurfer's
        #       (tkRAS)     (Talairach)      talairach.xfm read by MNE's
        #                                    read_ras_mni_t()
        #
        #   MNI305       →  MNI152           fixed published transform from
        #                                    FreeSurfer documentation (see
        #                                    surfer.nmr.mgh.harvard.edu/
        #                                    fswiki/CoordinateSystems)
        #
        # One centroid source per parcel is retained, giving a low-
        # dimensional source space suitable for TRF / connectivity.

        # --- Load / fetch the atlas ---------------------------------
        if os.path.isfile(vol_parc_atlas):
            vol_parc_path = vol_parc_atlas
            vol_parc_name = os.path.basename(vol_parc_path).replace('.nii.gz', '').replace('.nii', '')
            atlas_img = nib.load(vol_parc_path)
            atlas_labels = None
        else:
            from nilearn import datasets as ni_datasets
            if vol_parc_atlas == 'aal':
                atlas = ni_datasets.fetch_atlas_aal()
                vol_parc_path = atlas['maps']
                atlas_labels = atlas['labels']
            elif vol_parc_atlas == 'destrieux':
                atlas = ni_datasets.fetch_atlas_destrieux_2009()
                vol_parc_path = atlas['maps']
                atlas_labels = atlas['labels']
            elif vol_parc_atlas == 'harvard_oxford':
                atlas = ni_datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
                vol_parc_path = atlas['maps']
                atlas_labels = atlas['labels']
            elif vol_parc_atlas == 'schaefer':
                atlas = ni_datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
                vol_parc_path = atlas['maps']
                atlas_labels = atlas['labels']
            else:
                raise ValueError(
                    f'Unknown vol_parc_atlas "{vol_parc_atlas}". '
                    f'Use "aal", "destrieux", "harvard_oxford", "schaefer", '
                    f'or provide a path to a .nii.gz file.'
                )
            vol_parc_name = vol_parc_atlas
            atlas_img = nib.load(vol_parc_path)

        atlas_data = np.asarray(atlas_img.dataobj).astype(int)  # parcel IDs are integers
        atlas_affine = atlas_img.affine            # voxel -> MNI152 mm
        inv_atlas_affine = np.linalg.inv(atlas_affine)  # MNI152 mm -> voxel

        if atlas_labels is not None:
            print(f'Atlas "{vol_parc_name}" loaded: {len(atlas_labels)} labels, '
                  f'volume shape {atlas_data.shape}')

        # --- Full volume source space (needed to define candidate vertices) ---
        fname_src_vol = (paths.sources_path + subject_code
                         + f'/{subject_code}_volume_{spacing}_{int(pos)}-src.fif')
        try:
            src_vol = mne.read_source_spaces(fname_src_vol)
        except:
            src_vol = mne.setup_volume_source_space(
                subject=subject_code, subjects_dir=subjects_dir,
                bem=bem, pos=pos, sphere_units='m', add_interpolator=True
            )
            mne.write_source_spaces(fname_src_vol, src_vol, overwrite=True)

        # --- Map source positions to MNI152 and assign parcels ----------
        inuse_mask = src_vol[0]['inuse'].astype(bool)
        src_rr = src_vol[0]['rr'][inuse_mask]  # (n_src, 3) metres, surface-RAS

        # Step 1: surface-RAS → MNI305 (per-subject, from FreeSurfer's
        #         talairach.xfm via MNE's read_ras_mni_t)
        # NOTE: read_ras_mni_t returns a mm-based transform (same as the
        #       xfm file). MNE's own vertex_to_mni converts to mm before
        #       applying it. Source positions must be in mm here.
        ras_mni_t = read_ras_mni_t(subject_code, subjects_dir)
        src_rr_mm = src_rr * 1000                          # m → mm
        src_mni305_mm = apply_trans(ras_mni_t, src_rr_mm)  # MNI305, mm

        # Step 2: MNI305 → MNI152 (fixed coordinate-system relationship)
        # Published by FreeSurfer:
        # https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems
        # This is a known constant — it does NOT vary per subject.
        mni305_to_mni152 = np.array([
            [ 0.9975, -0.0073,  0.0176, -0.0429],
            [ 0.0146,  1.0009, -0.0024,  1.5496],
            [-0.0130, -0.0093,  0.9971,  1.1840],
            [ 0.0000,  0.0000,  0.0000,  1.0000],
        ])  # operates on mm
        src_mni152_mm = apply_trans(mni305_to_mni152, src_mni305_mm)

        # Step 3: MNI152 mm → atlas voxel indices
        src_vox = apply_trans(inv_atlas_affine, src_mni152_mm)
        src_vox_idx = np.round(src_vox).astype(int)
        for dim in range(3):
            src_vox_idx[:, dim] = np.clip(src_vox_idx[:, dim], 0, atlas_data.shape[dim] - 1)

        parcel_ids = atlas_data[src_vox_idx[:, 0], src_vox_idx[:, 1], src_vox_idx[:, 2]]

        # Keep only non-background parcels
        unique_parcels = np.unique(parcel_ids)
        unique_parcels = unique_parcels[unique_parcels != 0]

        if len(unique_parcels) == 0:
            raise RuntimeError(
                f'No source points fell inside any atlas parcel for {subject_code}. '
                f'Check the coordinate transform (print output above). '
                f'Expected MNI152 coords roughly in range x=[-80,80], y=[-120,80], z=[-60,90] mm.'
            )

        # Find the source closest to each parcel's centroid
        centroid_global_indices = []
        for p_id in unique_parcels:
            mask = parcel_ids == p_id
            parcel_coords = src_rr[mask]
            centroid = parcel_coords.mean(axis=0)
            dists = np.linalg.norm(parcel_coords - centroid, axis=1)
            centroid_global_indices.append(int(np.where(mask)[0][np.argmin(dists)]))

        # Map back from "inuse" indices to full rr-array indices
        inuse_indices = np.where(inuse_mask)[0]
        centroid_rr_indices = inuse_indices[np.array(centroid_global_indices, dtype=int)]

        # --- Restricted volume source space (one source per parcel) -----
        fname_src = (paths.sources_path + subject_code
                     + f'/{subject_code}_vol_parcellation_{vol_parc_name}-src.fif')
        try:
            src = mne.read_source_spaces(fname_src)
        except:
            src = src_vol.copy()
            src[0]['inuse'][:] = 0
            src[0]['inuse'][centroid_rr_indices] = 1
            src[0]['nuse'] = len(centroid_rr_indices)
            src[0]['vertno'] = np.sort(centroid_rr_indices)

            n_sources = len(centroid_rr_indices)
            print(f'Volume parcellation source space: {n_sources} sources '
                  f'from {len(unique_parcels)} parcels ({vol_parc_name})')
            mne.write_source_spaces(fname_src, src, overwrite=True)

        # --- Forward model ----------------------------------------------
        if subject_id != 'fsaverage':
            fname_fwd = (sources_path_subject
                         + f'/{subject_code}_{meg_params["data_type"]}_chs{chs_id}'
                           f'_vol_parcellation_{vol_parc_name}-fwd.fif')

            fwd = mne.make_forward_solution(
                meg_data.info, trans=trans_path, src=src, bem=bem
            )
            mne.write_forward_solution(fname_fwd, fwd, overwrite=True)

