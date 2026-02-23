import os
import mne
import numpy as np
import functions_general
import paths
import load
import setup
import pandas as pd
import plot_general

# Load experiment info
exp_info = setup.exp_info()


# --------- Setup ---------#
task = 'DA2'
# Define surface or volume source space
chs_id = 'mag_z'
surf_vol = 'surface'
force_fsaverage = True
spacing = 'ico4'
pos = 10
pick_ori = None # 'normal' | 'max-power' | None
depth = None

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
            trans_path = os.path.join(subjects_dir, subject_code, 'bem', f'{subject_code}-trans.fif')
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
                    trans_path = os.path.join(paths.sources_path, subject_code, f'{subject_code}-trans.fif')
                    trans = mne.read_trans(trans_path)
                    print('Distance from head origin to MEG origin: %0.1f mm'
                          % (1000 * np.linalg.norm(meg_data.info['dev_head_t']['trans'][:3, 3])))
                    print('Distance from head origin to MRI origin: %0.1f mm'
                          % (1000 * np.linalg.norm(trans['trans'][:3, 3])))
                except:
                    # Load digitalization file
                    dig_path_subject = dig_path + subject_id
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
                                                                rpa=rpa.values.ravel(), hsp=pos_array,
                                                                coord_frame='unknown')

                    # Make info object
                    dig_info = meg_data.pick('meg').info.copy()
                    dig_info.set_montage(montage=dig_montage)

                    # Save raw instance with info
                    info_raw = mne.io.RawArray(np.zeros((dig_info['nchan'], 1)), dig_info)
                    dig_info_path = dig_path_subject + '/info_raw.fif'
                    info_raw.save(dig_info_path, overwrite=True)

                    # Align and save fiducials and transformation files to FreeSurfer/subject/bem folder
                    mne.gui.coregistration(subject=subject_id, subjects_dir=subjects_dir, inst=None,
                                           block=True)

            # If subject has no MRI data
            else:
                subject_code = 'fsaverage'
                dig = False
                # Check mean distances if already run transformation
                trans_path = os.path.join(subjects_dir, subject_code, 'bem', f'{subject_code}-trans.fif')
                trans = mne.read_trans(trans_path)
                print('Distance from head origin to MEG origin: %0.1f mm'
                      % (1000 * np.linalg.norm(meg_data.info['dev_head_t']['trans'][:3, 3])))
                print('Distance from head origin to MRI origin: %0.1f mm'
                      % (1000 * np.linalg.norm(trans['trans'][:3, 3])))
    if visualize_alignment:
        # Load subject
        subject = setup.subject(subject_id=subject_id)
        plot_general.mri_meg_alignment(subject=subject, subject_code=subject_code, dig=dig, subjects_dir=subjects_dir)

    else:
        subject_code = 'fsaverage'
        dig = False

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
