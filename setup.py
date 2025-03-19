import paths
import mne
import pandas as pd
import pathlib
import os


class exp_info:
    """
    Class containing the experiment information.

    Attributes
    ----------
    ctf_path : str
        Path to the MEG data.
    et_path : str
        Path to the Eye-Tracker data.
    mri_path : str
        Path to the MRI data.
    opt_path : str
        Path to the Digitalization data.
    subjects_ids : list
        List of subject IDs.
    bad_channels : dict
        Dictionary of subject IDs and their corresponding bad channels.
    screen_distance : dict
        Dictionary of subject IDs and their distance to the screen during the experiment.
    screen_size : dict
        Dictionary of subject IDs and their screen size.
    group : dict
        Dictionary of subject IDs and their group (balanced or counterbalanced).
    et_channel_names : dict
        Dictionary of tracked eyes and their corresponding channel names.
    tracked_eye : dict
        Dictionary of subject IDs and their tracked eye.
    no_pupil_subjects : list
        List of subjects with missing pupil data.
    trig_ch : str
        Trigger channel name.
    button_ch : str
        Buttons channel name.
    buttons_ch_map : dict
        Map of button values to colors.
    buttons_pc_map : dict
        Map of button values to colors (PC version).
    DAC_delay : int
        DAC delay in milliseconds.
    noise_recordings : list
        List of background noise recordings date IDs.
    empty_room_recording : dict
        Dictionary of subject IDs and their associated background noise.
    background_line_noise_freqs : dict
        Dictionary of noise recording dates and their line noise frequencies.
    line_noise_freqs : dict
        Dictionary of subject IDs and their line noise frequencies.
    """

    def __init__(self):
        # Define opm data path and files path
        self.opm_path = paths.opm_path
        self.et_path = paths.et_path
        self.mri_path = paths.mri_path
        self.opt_path = paths.opt_path

        # Select subject
        self.subjects_ids = ['11074',
                             '11766',
                             '13229',
                             '13703',
                             '14446',
                             '15463',
                             '15519',
                             '16589',
                             '16659'
                             ]

        # Subjects bad channels
        self.bad_channels = {'11766': ['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11', 'A1','A3','A5','A7','A8','A10','A12','A14','A16']
                             }

        # Taken from Driving Experiment info v0_1
        self.exp_times = {'11074': {'cf_start': 59, 'cf_end': 172, 'da_start': 172, 'da_end': 535},
                          '11766': {'cf_start': 49, 'cf_end': 161, 'da_start': 161, 'da_end': 526},
                          '13229': {'cf_start': 55, 'cf_end': 168, 'da_start': 161, 'da_end': 546},
                          '13703': {'cf_start': 54, 'cf_end': 168, 'da_start': 161, 'da_end': 532},
                          '14446': {'cf_start': 50, 'cf_end': 164, 'da_start': 164, 'da_end': 528},
                          '15463': {'cf_start': 57, 'cf_end': 171, 'da_start': 171, 'da_end': 536},
                          '15519': {'cf_start': 59, 'cf_end': 172, 'da_start': 172, 'da_end': 546},
                          '16589': {'cf_start': 49, 'cf_end': 161, 'da_start': 161, 'da_end': 525},
                          '16659': {'cf_start': 70, 'cf_end': 183, 'da_start': 183, 'da_end': 581},
                          }

        # Dataframe containing DA symbols start times for each subject
        self.da_times = {key: pd.read_csv(paths.exp_path + f'da_time_{key}.csv', names=['DA times']) for key in self.subjects_ids}

        # Distance to the screen during the experiment (Fake info)
        self.screen_distance = {'11074': 68,
                                '11766': 68,
                                '13229': 68,
                                '13703': 68,
                                '14446': 68,
                                '15463': 68,
                                '15519': 68,
                                '16589': 68,
                                '16659': 68
                                }

        # Screen width (Fake info)
        self.screen_size = {'11074': 34,
                            '11766': 34,
                            '13229': 34,
                            '13703': 34,
                            '14446': 34,
                            '15463': 34,
                            '15519': 34,
                            '16589': 34,
                            '16659': 34
                            }

        # Subjects groups (Fake info)
        self.group = {'11074': 'counterbalanced',
                      '11766': 'counterbalanced',
                      '13229': 'counterbalanced',
                      '13703': 'counterbalanced',
                      '14446': 'counterbalanced',
                      '15463': 'counterbalanced',
                      '15519': 'counterbalanced',
                      '16589': 'counterbalanced',
                      '16659': 'counterbalanced'
                      }

        # Duration of the DA in seconds
        self.DA_duration = 4.5

        # Tracked eye (Fake info)
        self.tracked_eye = {'11074': 'left',
                            '11766': 'left',
                            '13229': 'left',
                            '13703': 'left',
                            '14446': 'left',
                            '15463': 'left',
                            '15519': 'left',
                            '16589': 'left',
                            '16659': 'left'
                            }

        # ET channels name [Gaze x, Gaze y, Pupils] (Fake info)
        self.et_channel_names = {'11074': ['UADC013-4123', 'UADC015-4123', 'UADC016-4123'],
                                 '11766': ['UADC013-4123', 'UADC015-4123', 'UADC016-4123'],
                                 '13229': ['UADC013-4123', 'UADC015-4123', 'UADC016-4123'],
                                 '13703': ['UADC013-4123', 'UADC015-4123', 'UADC016-4123'],
                                 '14446': ['UADC013-4123', 'UADC015-4123', 'UADC016-4123'],
                                 '15463': ['UADC013-4123', 'UADC015-4123', 'UADC016-4123'],
                                 '15519': ['UADC013-4123', 'UADC015-4123', 'UADC016-4123'],
                                 '16589': ['UADC013-4123', 'UADC015-4123', 'UADC016-4123'],
                                 '16659': ['UADC013-4123', 'UADC015-4123', 'UADC016-4123']
                                 }

        # Trigger channel name (Fake info)
        self.trig_ch = 'UPPT002'
        self.alt_trig_ch = 'UPPT001'

        # Buttons channel name (Fake info)
        self.button_ch = 'UPPT001'

        # Buttons values to colors map
        self.buttons_ch_map = {1: 'blue', 2: 'yellow', 4: 'green', 8: 'red', 'blue': 1, 'yellow': 2, 'green': 4, 'red': 8}
        self.buttons_pc_map = {1: 'blue', 2: 'yellow', 3: 'green', 4: 'red', 'blue': 1, 'yellow': 2, 'green': 3, 'red': 4}

        # DAC delay (in ms)
        self.DAC_delay = 10

        # Notch filter line noise frequencies (Fake info)
        self.line_noise_freqs = {'11074': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '11766': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '13229': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '13703': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '14446': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '15463': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '15519': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '16589': (50, 57, 100, 109, 150, 200, 250, 300),
                                 '16659': (50, 57, 100, 109, 150, 200, 250, 300)
                                 }


class analysis_parameters:
    """
    Class containing the analysis parameters.

    Attributes
    ----------
    ctf_path: str
        Path to the MEG data.
    et_path: str
        Path to the Eye-Tracker data.
    mri_path: str
        Path to the MRI data.
    opt_path: str
        Path to the Digitalization data.
    subjects_ids: list
        List of subject's id.
    subjects_bad_channels: list
        List of subject's bad channels.
    subjects_groups: list
        List of subject's group
    missing_bh_subjects: list
        List of subject's ids missing behavioural data.
    trials_loop_subjects: list
        List of subject;s ids for subjects that took the firts version of the experiment.
    """

    def __init__(self):
        # Samples drop at begining of missing pupils signal
        self.start_interval_samples = 24

        # Samples drop at end of missing pupils signal
        self.end_interval_samples = 24

        # Pupil size threshold to consider missing signal (Fake info)
        self.pupil_thresh = {'11074': -2.6,
                             '11766': -2.6,
                             '13229': -2.6,
                             '13703': -2.6,
                             '14446': -2.6,
                             '15463': -2.6,
                             '15519': -2.6,
                             '16589': -2.6,
                             '16659': -2.6
                             }

        # Et samples shift for ET-MEG alignment
        self.et_samples_shift = {}

        # Trial reject parameter based on MEG peak to peak amplitude (Fake info)
        self.reject_amp = {'11074': 5e-12,
                           '11766': 5e-12,
                           '13229': 5e-12,
                           '13703': 5e-12,
                           '14446': 5e-12,
                           '15463': 5e-12,
                           '15519': 5e-12,
                           '16589': 5e-12,
                           '16659': 5e-12
                           }


class subject():
    """
    Class containing subjects data and analysis parameters.

    Parameters
    ----------
    subject: {'int', 'str'}, default=None
        Subject id (str) or number (int). If None, takes the first subject.

    Attributes
    ----------
    bad_channels: list
        List of bad channels.
    ctf_path: str
        Path to the MEG data.
    et_path: str
        Path to the Eye-Tracker data.
    mri_path: str
        Path to the MRI data.
    opt_path: str
        Path to the Digitalization data.
    subject_id: str
        Subject id.
    """

    def __init__(self, subject_id=None):

        # Select 1st subject by default
        if subject_id == None:
            self.subject_id = exp_info().subjects_ids[0]
        # Select subject by index
        elif type(subject_id) == int:
            self.subject_id = exp_info().subjects_ids[subject_id]
        # Select subject by id
        elif type(subject_id) == str and (subject_id in exp_info().subjects_ids):
            self.subject_id = subject_id
        else:
            print('Subject not found')

        # Get attributes from experiment info
        exp_info_att = exp_info().__dict__.keys()
        for general_att in exp_info_att:
            att = getattr(exp_info(), general_att)
            if type(att) == dict:
                try:
                    # If subject_id in dictionary keys, get attribute
                    att_value = att[self.subject_id]
                    setattr(self, general_att, att_value)
                except:
                    pass

        # Get preprocessing and general configurations
        self.params = self.params(params=analysis_parameters(), subject_id=self.subject_id)

    # Subject's parameters and configuration
    class params:

        def __init__(self, params, subject_id):
            # Get attributes and get data for corresponding subject
            attributes = params.__dict__.keys()

            # Iterate over attributes and get data for corresponding subject
            for att_name in attributes:
                att = getattr(params, att_name)
                if type(att) == dict:
                    try:
                        # If subject_id in dictionary keys, get attribute, else pass
                        att_value = att[subject_id]
                        setattr(self, att_name, att_value)
                    except:
                        pass
                else:
                    # If attribute is general for all subjects, get attribute
                    att_value = att
                    setattr(self, att_name, att_value)

    # Raw MEG data
    def load_raw_meg_data(self):
        """
        MEG data for parent subject as Raw instance of MNE.
        """

        print('\nLoading Raw MEG data')
        # Get subject path
        subj_path = pathlib.Path(os.path.join(paths.opm_path, self.subject_id))
        ds_files = list(subj_path.glob('*{}*.ds'.format(self.subject_id)))
        ds_files.sort()

        # Load sesions
        # If more than 1 session concatenate all data to one raw data
        if len(ds_files) > 1:
            raws_list = []
            for i in range(len(ds_files)):
                raw = mne.io.read_raw_ctf(ds_files[i], system_clock='ignore')
                raws_list.append(raw)
            # MEG data structure
            raw = mne.io.concatenate_raws(raws_list, on_mismatch='ignore')

            # Set dev <-> head transformation from optimal head localization
            raw.info['dev_head_t'] = raws_list[self.params.general.head_loc_idx].info['dev_head_t']

        # If only one session return that session as whole raw data
        elif len(ds_files) == 1:
            raw = mne.io.read_raw_ctf(ds_files[0], system_clock='ignore')

        # Missing data
        else:
            raise ValueError('No .ds files found in subject directory: {}'.format(subj_path))

        return raw

    # MEG data
    def load_preproc_meg_data(self, preload=False):
        """
        Preprocessed MEG data for parent subject as raw instance of MNE.
        """

        # Subject preprocessed data path
        file_path = pathlib.Path(os.path.join(paths.preproc_path, self.subject_id, f'Subject_{self.subject_id}_meg.fif'))

        # Try to load preprocessed data
        try:
            print('\nLoading Preprocessed MEG data')
            meg_data = mne.io.read_raw_fif(file_path, preload=preload)
        except:
            raise ValueError(f'No previous preprocessed data found for subject {self.subject_id}')

        return meg_data

    # ICA MEG data
    def load_ica_meg_data(self, preload=False):
        """
        ICA MEG data for parent subject as Raw instance of MNE.
        """

        # Subject ICA data path
        file_path = pathlib.Path(os.path.join(paths.ica_path, self.subject_id, f'Subject_{self.subject_id}_ICA.fif'))

        # Try to load ica data
        try:
            print(f'Loading ICA data for subject {self.subject_id}')
            # Load data
            ica_data = mne.io.read_raw_fif(file_path, preload=preload)

        except:
            raise ValueError(f'No previous ica data found for subject {self.subject_id}')

        return ica_data


   # Behavioural data
    def load_raw_bh_data(self):
        """
        Behavioural data for parent subject as pandas DataFrames.
        """
        # Get subject path
        raise ('Update paths to use csv within ET data.')
        subj_path = pathlib.Path(os.path.join(paths.eh_path, self.subject_id))
        bh_file = list(subj_path.glob('*.csv'.format(self.subject_id)))[0]

        # Load DataFrame
        df = pd.read_csv(bh_file)

        return df