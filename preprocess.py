import os
import mne
import setup
import paths
import load
import functions_preproc


meg_params = {'data_type': 'ICA'}

# Load experiment info
exp_info = setup.exp_info()

for subject_id in exp_info.subjects_ids:
    print(f'Processing subject {subject_id}')

    subject = setup.subject(subject_id=subject_id)

    # Load MEG
    meg_data = load.meg(subject_id=subject_id, meg_params=meg_params)

    stim_data = meg_data.copy().pick('stim')

    # Get ET channels from MEG
    print('\nGetting ET channels data from MEG')
    et_channels_meg = stim_data.get_data(picks=exp_info.et_channel_names[subject_id])

    # ---------------- Blinks removal ----------------#
    # Define intervals around blinks to also fill with nan. Due to conversion noise from square signal
    et_channels_meg = functions_preproc.blinks_to_nan(meg_data=meg_data, et_channels_meg=et_channels_meg)

    # ---------------- Remove saccades and fixations annotations ---------------- #
    meg_data = functions_preproc.remove_annotations(meg_data=meg_data, subject=subject, exp_info=exp_info)

    # ---------------- Fixations and saccades detection ----------------#
    fixations, saccades, pursuits = functions_preproc.fixations_saccades_detection(meg_data=meg_data,
                                                                                   et_channels_meg=et_channels_meg,
                                                                                   subject=subject,
                                                                                   exp_info=exp_info,
                                                                                   force_run=True)

    # ---------------- Saccades classification ----------------#
    saccades, subject = functions_preproc.saccades_classification(subject=subject, saccades=saccades, meg_data=meg_data)

    # ---------------- Fixations classification ----------------#
    fixations = functions_preproc.fixations_classification(df=fixations, saccades=saccades, meg_data=meg_data, et_channels_meg=et_channels_meg, title='fixations')

    # ---------------- Pursuits classification ----------------#
    pursuits = functions_preproc.fixations_classification(df=pursuits, saccades=saccades, meg_data=meg_data, et_channels_meg=et_channels_meg, title='pursuits')

    # ---------------- Interpolate bad channels ----------------#
    # meg_data = functions_preproc.interpolate_bad_channels(subject_id, meg_data, exp_info)

    # ---------------- Add scaled ET channels back to MEG data ----------------#
    # meg_data = functions_preproc.add_et_channels(subject_id=subject_id, meg_data=meg_data, et_channels_meg=et_channels_meg, exp_info=exp_info)

    # ---------------- Save preprocessed data ----------------#
    functions_preproc.save(meg_data=meg_data, subject_id=subject_id, fixations=fixations, saccades=saccades, pursuits=pursuits, task='DA')
