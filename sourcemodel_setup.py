import os
import mne
import numpy as np
import paths
import load
import setup

# Load experiment info
exp_info = setup.exp_info()


# --------- Setup ---------#
task = 'DA'
# Define surface or volume source space
surf_vol = 'surface'
force_fsaverage = False
ico = 4
spacing = 10.
pick_ori = None
depth = None

# Define Subjects_dir as Freesurfer output folder
mri_path = paths.mri_path
subjects_dir = os.path.join(mri_path, 'freesurfer')
os.environ["SUBJECTS_DIR"] = subjects_dir

# Digitalization data path
dig_path = paths.opt_path


# --------- Coregistration ---------#

# Iterate over subjects
for subject_id in exp_info.subjects_ids + ['fsaverage']:

    if subject_id != 'fsaverage':
        meg_data = load.ica_data(subject_id=subject_id, task=task)

        if force_fsaverage:
            subject_code = 'fsaverage'
            # Check mean distances if already run transformation
            trans_path = os.path.join(subjects_dir, subject_code, 'bem', f'{subject_code}-trans.fif')
            trans = mne.read_trans(trans_path)
            print('Distance from head origin to MEG origin: %0.1f mm'
                  % (1000 * np.linalg.norm(meg_data.info['dev_head_t']['trans'][:3, 3])))
            print('Distance from head origin to MRI origin: %0.1f mm'
                  % (1000 * np.linalg.norm(trans['trans'][:3, 3])))

        else:
            subject_code = subject_id
            # Check if subject has MRI data
            try:
                # Check mean distances if already run transformation
                trans_path = os.path.join(paths.sources_path, subject_code, f'{subject_code}-trans.fif')
                trans = mne.read_trans(trans_path)
                print('Distance from head origin to MEG origin: %0.1f mm'
                      % (1000 * np.linalg.norm(meg_data.info['dev_head_t']['trans'][:3, 3])))
                print('Distance from head origin to MRI origin: %0.1f mm'
                      % (1000 * np.linalg.norm(trans['trans'][:3, 3])))

            # If subject has no MRI data
            except:
                subject_code = 'fsaverage'
                # Check mean distances if already run transformation
                trans_path = os.path.join(subjects_dir, subject_code, 'bem', f'{subject_code}-trans.fif')
                trans = mne.read_trans(trans_path)
                print('Distance from head origin to MEG origin: %0.1f mm'
                      % (1000 * np.linalg.norm(meg_data.info['dev_head_t']['trans'][:3, 3])))
                print('Distance from head origin to MRI origin: %0.1f mm'
                      % (1000 * np.linalg.norm(trans['trans'][:3, 3])))
    else:
        subject_code = 'fsaverage'

    # --------- Bem model ---------#
    # Source data and models path
    sources_path_subject = paths.sources_path + subject_id
    os.makedirs(sources_path_subject, exist_ok=True)

    fname_bem = paths.sources_path + subject_code + f'/{subject_code}_bem_ico{ico}-sol.fif'
    try:
        # Load
        bem = mne.read_bem_solution(fname_bem)
    except:
        # Compute
        model = mne.make_bem_model(subject=subject_code, ico=ico, conductivity=[0.3], subjects_dir=subjects_dir)
        bem = mne.make_bem_solution(model)
        # Save
        mne.write_bem_solution(fname_bem, bem, overwrite=True)

    # --------- Source space, forward model and inverse operator ---------#
    if surf_vol == 'volume':
        # Volume
        # Source model
        fname_src = paths.sources_path + subject_code + f'/{subject_code}_volume_ico{ico}_{int(spacing)}-src.fif'
        try:
            # Load
            src = mne.read_source_spaces(fname_src)
        except:
            # Compute
            src = mne.setup_volume_source_space(subject=subject_code, subjects_dir=subjects_dir, bem=bem, pos=spacing,
                                                sphere_units='m', add_interpolator=True)
            # Save
            mne.write_source_spaces(fname_src, src, overwrite=True)

        # Forward model
        if subject_id != 'fsaverage':
            fname_fwd = sources_path_subject + f'/{subject_code}_volume_ico{ico}_{int(spacing)}-fwd.fif'
            try:
                # Load
                fwd = mne.read_forward_solution(fname=fname_fwd)
            except:
                # Compute
                fwd = mne.make_forward_solution(meg_data.info, trans=trans_path, src=src, bem=bem)
                mne.write_forward_solution(fname_fwd, fwd, overwrite=True)

    elif surf_vol == 'surface':
        # Source model
        fname_src = paths.sources_path + subject_code + f'/{subject_code}_surface_ico{ico}-src.fif'
        try:
            # Load
            src = mne.read_source_spaces(fname_src)
        except:
            # Compute
            src = mne.setup_source_space(subject=subject_code, spacing=f'ico{ico}', subjects_dir=subjects_dir)
            # Save
            mne.write_source_spaces(fname_src, src, overwrite=True)

        # Forward model
        if subject_id != 'fsaverage':
            fname_fwd = sources_path_subject + f'/{subject_code}_surface_ico{ico}-fwd.fif'
            try:
                # Load
                fwd = mne.read_forward_solution(fname=fname_fwd)
            except:
                # Compute
                fwd = mne.make_forward_solution(meg_data.info, trans=trans_path, src=src, bem=bem)
                mne.write_forward_solution(fname_fwd, fwd, overwrite=True)

    elif surf_vol == 'mixed':
        fname_src_mix = paths.sources_path + subject_code + f'/{subject_code}_mixed_ico{ico}_{int(spacing)}-src.fif'
        try:
            # Load
            src_surf = mne.read_source_spaces(fname_src_surf)
        except:
            # Mixed
            # Surface source model
            fname_src_surf = paths.sources_path + subject_code + f'/{subject_code}_surface_ico{ico}-src.fif'
            try:
                # Load
                src_surf = mne.read_source_spaces(fname_src_surf)
            except:
                # Compute
                src_surf = mne.setup_source_space(subject=subject_code, spacing=f'ico{ico}', subjects_dir=subjects_dir)
                # Save
                mne.write_source_spaces(fname_src_surf, src_surf, overwrite=True)

            # Volume source model
            fname_src_vol = paths.sources_path + subject_code + f'/{subject_code}_volume_ico{ico}_{int(spacing)}-src.fif'
            try:
                # Load
                src_vol = mne.read_source_spaces(fname_src_vol)
            except:
                # Compute
                src_vol = mne.setup_volume_source_space(subject=subject_code, subjects_dir=subjects_dir, bem=bem, pos=spacing, sphere_units='m', add_interpolator=True)
                # Save
                mne.write_source_spaces(fname_src_vol, src_vol, overwrite=True)

            # Mixed source space
            src = src_surf + src_vol
            # Save
            mne.write_source_spaces(fname_src_mix, src, overwrite=True)

        # Forward model
        if subject_id != 'fsaverage':
            fwd = mne.make_forward_solution(meg_data.info, trans=trans_path, src=src, bem=bem)
            fname_fwd = sources_path_subject + f'/{subject_code}_mixed_ico{ico}_{int(spacing)}-fwd.fif'
            mne.write_forward_solution(fname_fwd, fwd, overwrite=True)
